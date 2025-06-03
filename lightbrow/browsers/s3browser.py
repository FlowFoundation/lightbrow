import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import dash 
from dash import dcc, html, Input, Output, State, callback, clientside_callback, no_update
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import pandas as pd
import pyperclip
from ..connectors.base import BaseConnector, FileItem, AccessLevel, AccessError
from ..search_engines.simple_seach import SimpleSearch

class S3Browser:
    """Main S3 Browser Dash application."""
    
    def __init__(self, bucket_connector_pairs: List[Tuple[str, BaseConnector]], max_depth: Optional[int] = None, debug: bool = False):
        # raise ValueError if bucket_connector_pairs wrong type
        if not isinstance(bucket_connector_pairs, list) or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in bucket_connector_pairs):
            raise ValueError("bucket_connector_pairs must be a list of tuples (bucket_name, BaseConnector instance).")
        if not all(isinstance(c, BaseConnector) for _, c in bucket_connector_pairs):
            raise ValueError("All items in bucket_connector_pairs must be tuples of (bucket_name, BaseConnector instance).")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None.")
        self.bucket_connector_pairs = bucket_connector_pairs
        self.max_depth = max_depth
        self.connectors = {b: c for b, c in bucket_connector_pairs} if bucket_connector_pairs else {}
        self.search_engines = {b: SimpleSearch(c) for b, c in bucket_connector_pairs} if bucket_connector_pairs else {}
        self.debug = debug
        self.app = dash.Dash(__name__, external_stylesheets=[
            "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
        ], suppress_callback_exceptions=True) 
        
        self.initial_current_paths = {}
        if self.bucket_connector_pairs:
            # Ensure initial paths end with a slash
            self.initial_current_paths = {
                b: (f"{c.default_prefix}{b}/" if not f"{c.default_prefix}{b}".endswith('/') else f"{c.default_prefix}{b}")
                for b, c in self.bucket_connector_pairs
            }
        
        self.notification_id_counter = 0 
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _get_next_notification_id(self) -> str:
        self.notification_id_counter += 1
        return f"notification-{self.notification_id_counter}"

    def _setup_layout(self):
        initial_bucket = self.bucket_connector_pairs[0][0] if self.bucket_connector_pairs else None
        bucket_options = [{"value": b, "label": b} for b, _ in self.bucket_connector_pairs]
        
        theme_toggle = dmc.Switch(
            offLabel=DashIconify(icon="radix-icons:sun", width=15, color=dmc.DEFAULT_THEME["colors"]["yellow"][8]),
            onLabel=DashIconify(icon="radix-icons:moon",width=15, color=dmc.DEFAULT_THEME["colors"]["yellow"][6]),
            id="color-scheme-toggle", persistence=True, color="grey",
        )   
        header_content = dmc.Group([
            dmc.Title("S3 Browser", order=3),
            dmc.Select(
                id="bucket-selector", data=bucket_options, value=initial_bucket, w=200,
                leftSection=DashIconify(icon="tabler:bucket"), disabled=not bool(initial_bucket)
            ),
            dmc.Popover(
                id="search-popover", withArrow=True, shadow="md", position="bottom-start", offset=5,
                # trapFocus=False, # Consider if needed
                # closeOnClickOutside=True, # Default, usually desired
                children=[
                    dmc.PopoverTarget(
                        dmc.TextInput(
                            id="search-input", placeholder="Search (e.g., *.txt)", w=600,
                            leftSection=DashIconify(icon="tabler:search"), 
                            rightSection=dmc.Loader(size="xs", id="search-loader", style={"display": "none"}),
                        )
                    ),
                    dmc.PopoverDropdown(
                        id="search-results-popover-dropdown",
                        style={"width": 600}, # Match search input width
                        children=[
                            dmc.ScrollArea(
                                id="search-results-area", h=500, 
                                children=[dmc.Text("Enter a search term.", c="dimmed", ta="center")]
                            )
                        ]
                    )
                ]
            ),
            theme_toggle,
        ], justify="space-between", p="md", style={"borderBottom": "1px solid var(--mantine-color-gray-3)"})

        # Define main layout items that will be overlaid
        main_layout_content_items = [
            dmc.Breadcrumbs(id="breadcrumbs-container", my="sm", children=[]),
            dmc.Divider(label="Browser", labelPosition="center"),
            dmc.ScrollArea(
                id="columns-scroll-area", 
                children=[dmc.Group(id="columns-container", gap="xs", align="stretch", wrap="nowrap", h="100%")], 
                type="auto", offsetScrollbars=True, style={"width": "100%", "flex": 1}
            ),
        ]

        # Wrapper for main content that will have the LoadingOverlay
        main_content_area_with_overlay = dmc.Box(
            style={"display": "flex", "flexDirection": "column", "height": "100%", "position": "relative"},
            children=[
                dmc.LoadingOverlay(
                    id="loading-overlay",
                    visible=False, # Controlled by callback
                    loaderProps={"type": "bars", "color": "blue", "size": "lg"}, # Styled loader
                    overlayProps={"radius": "sm", "blur": 2}, # Styled overlay
                    zIndex=1000, # Ensure it's on top of content
                ),
                *main_layout_content_items # Unpack the actual content items
            ]
        )

        footer_content = dmc.Group([
            
            dmc.Button("holder",id="current-path-display", style={"flexGrow": 2}, leftSection=DashIconify(icon="fluent:database-plug-connected-20-filled"), variant="outline", disabled=True, radius="md"),
            dmc.Group([
                dmc.Button("Copy Path", id="copy-path-button", size="xs", variant="subtle", leftSection=DashIconify(icon="tabler:copy")),
            
                dmc.Button("Up", id="up-directory-button", size="xs", variant="outline", leftSection=DashIconify(icon="tabler:arrow-up"))
            ], gap="xs")
        ], justify="space-between", p="xs", style={"borderTop": "1px solid var(--mantine-color-gray-3)"})

        self.app.layout = dmc.MantineProvider(
            theme={"fontFamily": "Inter, sans-serif", "primaryColor": "blue", "colorScheme": "light"},
            id="mantine-provider",
            children=[
                html.Div(id="notifications-container3", style={"position": "fixed", "top": "1rem", "right": "1rem", "zIndex": 3000}),
                html.Div(id="notifications-container", style={"position": "fixed", "top": "1rem", "right": "1rem", "zIndex": 2000}),
                dcc.Store(id="current-path-store", data=self.initial_current_paths),
                dcc.Store(id="selected-item-info-store", data=None),
                dcc.Store(id="error-store", data={"id": None, "message": None, "title": None}),
                dcc.Store(id="last-error-id-store", data=None),
                dmc.Modal(id="item-details-modal", title="Item Details", zIndex=10001, children=[html.Div(id="item-details-content")]),
                dmc.AppShell(
                    children=[
                        dmc.AppShellHeader(children=header_content),
                        dmc.AppShellMain(children=[main_content_area_with_overlay]), # Use the Box with overlay
                        dmc.AppShellFooter(children=footer_content)
                    ],
                    header={"height": 60}, padding="md"
                )
            ]
        )
        # connector = self.connectors.get(initial_bucket, None) 
        # if connector:
        #     connector.start_background_indexing(initial_bucket, self.max_depth) # Start indexing for the initial bucket

    def _setup_callbacks(self):
        clientside_callback(
            """(switchOn) => { document.documentElement.setAttribute('data-mantine-color-scheme', switchOn ? 'dark' : 'light'); return window.dash_clientside.no_update }""",
            Output("color-scheme-toggle", "id"), Input("color-scheme-toggle", "checked"),
        )

        @self.app.callback(
            [Output("columns-container", "children"),
             Output("breadcrumbs-container", "children"),
             Output("current-path-display", "children"),
             Output("loading-overlay", "visible"), 
             Output("error-store", "data")],
            [Input("current-path-store", "data"), Input("bucket-selector", "value")],
            prevent_initial_call=False 
        )
        def update_browser_view(current_paths_data, selected_bucket):
            if not selected_bucket: # No bucket selected
                if not self.bucket_connector_pairs: # No buckets configured at all
                    return [], [dmc.Text("No buckets configured.")], "No buckets.", False, no_update
                return [], [], "Select a bucket.", False, no_update # Prompt to select bucket

            # Attempt to get current path for the selected bucket
            current_path = current_paths_data.get(selected_bucket)
            connector = self.connectors.get(selected_bucket)

            # Handle missing connector or path
            if not connector:
                return [], [dmc.Text("Configuration error.")], "Error.", False, \
                       {"id": self._get_next_notification_id(), "title": "Connector Error", "message": f"No connector configured for bucket '{selected_bucket}'."}

            if not current_path: # If path is missing for this bucket, default to its root
                current_path = f"{connector.default_prefix}{selected_bucket}/"
                # It might be good to update current_paths_data here, but that requires another output and can create loops if not careful.
                # For now, we'll just use this default for this run. The display will show it.
                if self.debug: print(f"Warning: No current_path for {selected_bucket}, defaulting to {current_path}")


            # Ensure current_path (which should represent a directory for browser view) ends with a slash
            if not current_path.endswith('/'):
                current_path += '/'
            
            columns_children, breadcrumbs_ui = [], []
            error_to_show = no_update
            show_loading = True # Start by assuming we will show loading

            try:
                bucket_root_path = f"{connector.default_prefix}{selected_bucket}/"
                
                if not current_path.startswith(bucket_root_path):
                    if self.debug: print(f"Warning: current_path '{current_path}' for bucket '{selected_bucket}' doesn't start with its root '{bucket_root_path}'. Resetting.")
                    current_path = bucket_root_path 

                relative_path_from_bucket_root = current_path[len(bucket_root_path):]
                path_segments = [s for s in relative_path_from_bucket_root.strip('/').split('/') if s]

                accumulated_path_for_column = bucket_root_path
                # First column: root of the current bucket
                root_items = self._get_items_sync(connector, accumulated_path_for_column, selected_bucket)
                columns_children.append(self._create_column_ui(root_items, 0, accumulated_path_for_column, selected_bucket, current_path))

                # Subsequent columns for each segment in the path
                for i, segment in enumerate(path_segments):
                    accumulated_path_for_column += segment + "/"
                    segment_items = self._get_items_sync(connector, accumulated_path_for_column, selected_bucket)
                    columns_children.append(self._create_column_ui(segment_items, i + 1, accumulated_path_for_column, selected_bucket, current_path))
                
                breadcrumbs_ui = self._build_breadcrumbs(current_path, selected_bucket, connector.default_prefix)
                connector = self.connectors.get(selected_bucket) # Re-fetch connector in case it was updated
                connector.start_background_indexing(selected_bucket, self.max_depth) # Ensure indexing is running for the selected bucket
            
            except AccessError as e:
                if self.debug: print(f"AccessError in update_browser_view for {current_path}: {e.message}")
                error_to_show = {"id": self._get_next_notification_id(), "title": "Access Denied", "message": e.message}
                columns_children = [dmc.Paper(dmc.Text(f"Error: {e.message}", c="red", p="md"), w=280, p="xs", h="100%")]
                breadcrumbs_ui = self._build_breadcrumbs(current_path, selected_bucket, connector.default_prefix) # Still show breadcrumbs
            except Exception as e:
                if self.debug: print(f"Exception in update_browser_view for {current_path}: {str(e)}")
                import traceback; traceback.print_exc()
                error_to_show = {"id": self._get_next_notification_id(), "title": "Application Error", "message": f"Failed to display path: {str(e)}"}
                columns_children = [dmc.Paper(dmc.Text(f"Error: {str(e)}", c="red", p="md"), w=280, h="100%")]
                breadcrumbs_ui = [dmc.Text("Error building view")]
            finally:
                show_loading = False # Hide loading once processing is done or an error occurs
            
            return columns_children, breadcrumbs_ui, current_path, show_loading, error_to_show

        @self.app.callback(
            Output("current-path-store", "data", allow_duplicate=True),
            [Input({"type": "browser-item", "identity": dash.ALL, "bucket": dash.ALL}, "n_clicks")],
            [State("current-path-store", "data")],
            prevent_initial_call=True
        )
        def handle_item_navigation(n_clicks_all, current_paths_data):
            ctx = dash.callback_context
            if not ctx.triggered_id or not any(n_clicks_all): # Ensure there was a click
                return no_update

            clicked_item_id_dict = ctx.triggered_id
            item_s3_path = clicked_item_id_dict["identity"]
            item_bucket = clicked_item_id_dict["bucket"]
            
            connector = self.connectors.get(item_bucket)
            if not connector:
                if self.debug: print(f"Error: No connector for bucket {item_bucket} in handle_item_navigation.")
                # Optionally: return an error to error-store here
                return no_update

            item_info = self._get_item_info_sync(connector, item_s3_path, item_bucket)
            if not item_info: 
                if self.debug: print(f"Error: Could not get item info for {item_s3_path} in handle_item_navigation.")
                return no_update

            if item_info.is_directory and item_info.access_level != AccessLevel.NO_ACCESS:
                new_paths_data = current_paths_data.copy()
                new_path = item_s3_path if item_s3_path.endswith('/') else item_s3_path + '/'
                new_paths_data[item_bucket] = new_path
                return new_paths_data
            
            return no_update

        @self.app.callback(
            [Output("selected-item-info-store", "data", allow_duplicate=True),
             Output("item-details-modal", "opened", allow_duplicate=True),
             Output("error-store", "data", allow_duplicate=True)],
            [Input({"type": "browser-item", "identity": dash.ALL, "bucket": dash.ALL}, "n_clicks")],
            prevent_initial_call=True
        )
        def handle_item_modal(n_clicks_all): 
            ctx = dash.callback_context
            if not ctx.triggered_id or not any(n_clicks_all):
                return no_update, no_update, no_update

            clicked_item_id_dict = ctx.triggered_id
            item_s3_path = clicked_item_id_dict["identity"]
            item_bucket = clicked_item_id_dict["bucket"]
            
            connector = self.connectors.get(item_bucket)
            if not connector:
                return no_update, False, {"id": self._get_next_notification_id(), "title": "Error", "message": f"No connector for {item_bucket}."}

            item_info = self._get_item_info_sync(connector, item_s3_path, item_bucket)
            error_for_notification = no_update
            open_modal = False
            selected_item_data = no_update

            if not item_info:
                err_msg = connector.get_access_error(item_s3_path) or f"Info retrieval failed for {item_s3_path}"
                selected_item_data = {"error": err_msg} # Store error for modal content
                error_for_notification = {"id": self._get_next_notification_id(), "title": "Item Error", "message": err_msg}
            elif item_info.access_level == AccessLevel.NO_ACCESS:
                err_msg = item_info.access_error or f"Access denied: {item_s3_path}"
                selected_item_data = item_info.to_dict() # Store info for modal content (to show error)
                error_for_notification = {"id": self._get_next_notification_id(), "title": "Access Denied", "message": err_msg}
            elif not item_info.is_directory:  # Only open modal for accessible files
                selected_item_data = item_info.to_dict()
                open_modal = True
            else: # Accessible directory, store info but don't open modal
                selected_item_data = item_info.to_dict()
            
            return selected_item_data, open_modal, error_for_notification


        @self.app.callback(
            Output("item-details-content", "children"),
            Input("selected-item-info-store", "data"),
            prevent_initial_call=True
        )
        def update_modal_content(item_info_dict):
            if not item_info_dict: 
                return dmc.Text("No item selected.") 
            if "error" in item_info_dict and item_info_dict["error"]: 
                return dmc.Alert(item_info_dict["error"], title="Error Retrieving Item Details", color="red", withCloseButton=True)

            try:
                lm_iso = item_info_dict.get('last_modified')
                last_mod_dt = None
                if lm_iso:
                    try: last_mod_dt = datetime.fromisoformat(lm_iso.replace('Z', '+00:00'))
                    except ValueError: print(f"Warning: Could not parse last_modified date: {lm_iso}")

                acc_lvl_val = item_info_dict.get('access_level', AccessLevel.UNKNOWN.value)
                try: acc_lvl = AccessLevel(acc_lvl_val)
                except ValueError: acc_lvl = AccessLevel.UNKNOWN
                
                item = FileItem(
                    name=item_info_dict.get('name', 'N/A'), path=item_info_dict.get('path', 'N/A'), 
                    is_directory=item_info_dict.get('is_directory', False), size=item_info_dict.get('size'),
                    last_modified=last_mod_dt, access_level=acc_lvl, 
                    access_error=item_info_dict.get('access_error')
                )

                # Even for directories, if selected-item-info-store was updated, show some info.
                # Modal opening is controlled by handle_item_modal / handle_search_result_modal.
                details = [
                    dmc.Group([dmc.Text("Type:", fw=500), dmc.Text("Folder" if item.is_directory else "File")]),
                    dmc.Group([dmc.Text("Name:", fw=500), dmc.Text(item.name)]),
                    dmc.Group([dmc.Text("Path:", fw=500), dmc.Text(item.path, style={"wordBreak": "break-all"})]),
                ]
                if not item.is_directory:
                    details.append(dmc.Group([dmc.Text("Size:", fw=500), dmc.Text(self._format_size(item.size) if item.size is not None else "N/A")]))
                
                details.extend([
                    dmc.Group([dmc.Text("Modified:", fw=500), dmc.Text(item.last_modified.strftime("%Y-%m-%d %H:%M") if item.last_modified else "N/A")]),
                    dmc.Group([dmc.Text("Access:", fw=500), dmc.Badge(item.access_level.value, color="blue" if item.access_level == AccessLevel.FULL_ACCESS else "orange" if item.access_level == AccessLevel.READ_ONLY else "red" if item.access_level == AccessLevel.NO_ACCESS else "gray" )]),
                ])

                if item.access_error: 
                    details.append(dmc.Alert(item.access_error, title="Access Note", color="orange", withCloseButton=True))
                return dmc.Stack(details, gap="sm")
            except Exception as e:
                if self.debug: print(f"Error in update_modal_content: {str(e)}")
                return dmc.Alert(f"Error displaying details: {str(e)}", title="Display Error", color="red", withCloseButton=True)

        @self.app.callback(
            Output("current-path-store", "data", allow_duplicate=True),
            Input({"type": "breadcrumb-item", "path": dash.ALL, "bucket": dash.ALL}, "n_clicks"),
            State("current-path-store", "data"),
            prevent_initial_call=True
        )
        def handle_breadcrumb_click(n_clicks_all, current_paths_data):
            ctx = dash.callback_context
            if not ctx.triggered_id or not any(n_clicks_all): 
                return dash.no_update
            clicked_item_props = ctx.triggered_id
            path = clicked_item_props["path"]
            bucket = clicked_item_props["bucket"]
            
            if not path.endswith('/'): path += '/' # Breadcrumbs always point to directories

            new_paths_data = current_paths_data.copy()
            new_paths_data[bucket] = path
            return new_paths_data

        @self.app.callback(
            Output("current-path-store", "data", allow_duplicate=True),
            Input("up-directory-button", "n_clicks"),
            [State("current-path-store", "data"), State("bucket-selector", "value")],
            prevent_initial_call=True
        )
        def go_up_directory(n_clicks, current_paths_data, selected_bucket):
            if not n_clicks or not selected_bucket or selected_bucket not in current_paths_data: 
                return dash.no_update

            current_path = current_paths_data[selected_bucket]
            connector = self.connectors.get(selected_bucket)
            if not connector: 
                if self.debug: print(f"Error: No connector for bucket {selected_bucket} in go_up_directory.")
                return dash.no_update

            bucket_root_path = f"{connector.default_prefix}{selected_bucket}/"
            if not current_path.endswith('/'): current_path += '/'

            if not current_path.startswith(bucket_root_path) or current_path == bucket_root_path:
                return dash.no_update 

            relative_path_in_bucket = current_path[len(bucket_root_path):]
            if not relative_path_in_bucket.strip('/'): return dash.no_update

            segments = relative_path_in_bucket.strip('/').split('/')
            parent_relative_key = "/".join(segments[:-1]) + "/" if len(segments) > 1 else ""
            new_path = f"{bucket_root_path}{parent_relative_key}"
            
            new_paths_data = current_paths_data.copy()
            new_paths_data[selected_bucket] = new_path
            return new_paths_data

        @self.app.callback(
            Output("notifications-container3", "children"),
            Input("copy-path-button", "n_clicks"),
            State("current-path-display", "children"), # This is the displayed path string
            prevent_initial_call=True
        )
        def copy_current_path(n_clicks, current_path_to_copy):
            if n_clicks and isinstance(current_path_to_copy, str):
                pyperclip.copy(current_path_to_copy)
                # Optionally trigger a success notification via error-store (with a non-error title/color)
                return dmc.Notification(
                    title=f"Copied to clipboard",
                    autoClose=True,
                    action="show",
                    message=current_path_to_copy,
                    color="red",
                    position="bottom-right"

                )


        @self.app.callback(
            [Output("search-results-area", "children"), 
             Output("search-loader", "style"),
             Output("search-popover", "opened")],
            Input("search-input", "value"), # Debounced input
            [State("bucket-selector", "value"), State("current-path-store", "data")],
            prevent_initial_call=True
        )
        def handle_search(query_val, selected_bucket, current_paths_data):
            query_val_stripped = query_val.strip() if query_val else ""
            loader_style = {"display": "block"} # Show loader initially
            popover_opened_state = True # Default to open if query exists

            if not query_val_stripped:
                return dmc.Text("Enter search term.", c="dimmed", ta="center"), {"display": "none"}, False

            if not selected_bucket:
                return dmc.Text("Select bucket to search.", c="dimmed", ta="center"), {"display": "none"}, popover_opened_state

            connector = self.connectors.get(selected_bucket)
            search_engine = self.search_engines.get(selected_bucket)

            if not connector or not search_engine:
                return dmc.Text("Search unavailable for this bucket.", c="red", ta="center"), {"display": "none"}, popover_opened_state
            
            current_nav_path = current_paths_data.get(selected_bucket, f"{connector.default_prefix}{selected_bucket}/")
            _, search_prefix_key = connector._parse_s3_path(current_nav_path)
            
            try:
                results = search_engine.search(query_val_stripped, selected_bucket, search_prefix_key)
                if not results:
                    return dmc.Text(f"No results for '{query_val_stripped}'.", c="dimmed", ta="center"), {"display": "none"}, popover_opened_state

                result_items_ui = [
                    dmc.UnstyledButton(
                        id={"type": "search-result-item", "identity": item.path, "bucket": selected_bucket},
                        children=dmc.Stack([
                            DashIconify(icon="tabler:folder" if item.is_directory else "tabler:file", 
                                        color="blue" if item.is_directory else "gray", width=18),
                            dmc.Text(item.name, size="sm", truncate="end", maw=220),
                            dmc.Badge(
                                item.path,
                                variant="light",
                                color="#339af0",
                                size="md",
                                radius="xl",
                            ),
                        ], gap="xs"),
                        w="100%", p="xs", m="1px",
                        styles={"root": {"borderRadius": "var(--mantine-radius-sm)", "&:hover": {"backgroundColor": "var(--mantine-color-action-hover)"}}}
                    ) for item in results
                ]
                return dmc.Stack(result_items_ui, gap=0), {"display": "none"}, popover_opened_state
            except Exception as e:
                if self.debug: print(f"Error during search: {e}")
                return dmc.Text(f"Error during search: {e}", c="red", ta="center"), {"display": "none"}, popover_opened_state
            finally:
                # Loader style is set to {"display": "none"} by default in the success/no-result paths.
                # If an exception occurs before that, it might stay visible.
                # However, the rightSection of TextInput is where loader is.
                # The style output here is for the dmc.Loader component itself, not its container.
                pass


        @self.app.callback(
            [Output("current-path-store", "data", allow_duplicate=True),
             Output("search-input", "value", allow_duplicate=True), 
             Output("search-popover", "opened", allow_duplicate=True)],
            Input({"type": "search-result-item", "identity": dash.ALL, "bucket": dash.ALL}, "n_clicks"),
            [State("current-path-store", "data")],
            prevent_initial_call=True
        )
        def handle_search_result_navigation(n_clicks_all, current_paths_data):
            ctx = dash.callback_context
            if not ctx.triggered_id or not any(n_clicks_all): 
                return no_update, no_update, no_update
            
            clicked_search_item_id = ctx.triggered_id
            item_s3_path = clicked_search_item_id["identity"]
            item_bucket = clicked_search_item_id["bucket"]

            connector = self.connectors.get(item_bucket)
            if not connector: 
                return no_update, "", False # Clear input, close popover on error

            item_info = self._get_item_info_sync(connector, item_s3_path, item_bucket)
            if not item_info or item_info.access_level == AccessLevel.NO_ACCESS :
                # Error or no access, still clear input and close popover.
                # Notification of access denial should come from handle_search_result_modal
                return no_update, "", False 

            new_paths_data = current_paths_data.copy()
            if item_info.is_directory:
                new_path = item_s3_path if item_s3_path.endswith('/') else item_s3_path + '/'
                new_paths_data[item_bucket] = new_path
            else: 
                _parsed_bucket, key_part = connector._parse_s3_path(item_s3_path)
                parent_key = "/".join(key_part.strip('/').split('/')[:-1]) + "/" if '/' in key_part.strip('/') else ""
                parent_path = f"{connector.default_prefix}{item_bucket}/{parent_key}"
                new_paths_data[item_bucket] = parent_path
            
            return new_paths_data, "", False # Clear search input, close popover after successful navigation setup

        @self.app.callback(
            [Output("item-details-modal", "opened", allow_duplicate=True),
             Output("selected-item-info-store", "data", allow_duplicate=True),
             Output("error-store", "data", allow_duplicate=True)],
            Input({"type": "search-result-item", "identity": dash.ALL, "bucket": dash.ALL}, "n_clicks"),
            prevent_initial_call=True # Renamed from handle_search_result_modal_trigger for clarity
        )
        def handle_search_result_modal_opening(n_clicks_all): 
            ctx = dash.callback_context
            if not ctx.triggered_id or not any(n_clicks_all): 
                return no_update, no_update, no_update
            
            clicked_item_id_dict = ctx.triggered_id # Corrected variable name
            item_s3_path = clicked_item_id_dict["identity"]
            item_bucket = clicked_item_id_dict["bucket"]

            connector = self.connectors.get(item_bucket)
            error_for_notification = no_update
            open_modal = False
            selected_item_data = no_update

            if not connector:
                error_for_notification = {"id": self._get_next_notification_id(), "title": "Error", "message": f"No connector for {item_bucket}"}
                return False, no_update, error_for_notification

            item_info = self._get_item_info_sync(connector, item_s3_path, item_bucket)

            if not item_info:
                err_msg = connector.get_access_error(item_s3_path) or f"Info retrieval failed for {item_s3_path}"
                selected_item_data = {"error": err_msg}
                error_for_notification = {"id": self._get_next_notification_id(), "title": "Item Error", "message": err_msg}
            elif item_info.access_level == AccessLevel.NO_ACCESS:
                err_msg = item_info.access_error or f"Access denied: {item_s3_path}"
                selected_item_data = item_info.to_dict()
                error_for_notification = {"id": self._get_next_notification_id(), "title": "Access Denied", "message": err_msg}
            elif not item_info.is_directory: 
                selected_item_data = item_info.to_dict()
                open_modal = True
            else: # Directory clicked from search, store info but don't open modal (navigation handles it)
                 selected_item_data = item_info.to_dict()


            return open_modal, selected_item_data, error_for_notification

        @self.app.callback(
            [Output("notifications-container", "children"), Output("last-error-id-store", "data")],
            Input("error-store", "data"),
            [State("notifications-container", "children"), State("last-error-id-store", "data")],
            prevent_initial_call=True
        )
        def show_error_notification(error_data, existing_notifs, last_error_id):
            existing_notifs = existing_notifs or []
            if error_data and error_data.get("message") and error_data.get("id") and error_data.get("id") != last_error_id:
                title = error_data.get("title", "Notification")
                message = error_data.get("message")
                color = "red" if "error" in title.lower() or "denied" in title.lower() else "blue"
                new_notif = dmc.Notification(
                    id=f"notif-{error_data['id']}", title=title, 
                    message=dmc.Text(message,truncate="end",lineClamp=3), color=color,
                    icon=DashIconify(icon="tabler:alert-circle" if color == "red" else "tabler:info-circle"),
                    action="show", autoClose=7000,
                )
                return ([new_notif] + existing_notifs)[:5], error_data.get("id")
            return no_update, no_update

    def _get_items_sync(self, connector: BaseConnector, path: str, bucket_name: str) -> List[FileItem]:
        try:
            # asyncio.run can cause issues if an event loop is already running.
            # Submitting to executor which then runs asyncio.run in its own thread is safer.
            future = connector._executor.submit(asyncio.run, connector.list_items(path))
            return future.result(timeout=30) 
        except AccessError: raise 
        except asyncio.TimeoutError: raise AccessError(f"Timeout listing items for {path}", "ListTimeout")
        except Exception as e:
            if self.debug: print(f"Unhandled error in _get_items_sync for {path} ({bucket_name}): {e}")
            raise AccessError(f"Failed to list items for {path}: An unexpected error occurred.", "ListSyncError")

    def _get_item_info_sync(self, connector: BaseConnector, path: str, bucket_name: str) -> Optional[FileItem]:
        try:
            future = connector._executor.submit(asyncio.run, connector.get_item_info(path))
            return future.result(timeout=10)
        except AccessError as ae:
            if self.debug: print(f"AccessError in _get_item_info_sync for {path}: {ae.message}")
            name = path.strip('/').split('/')[-1] if path.strip('/') else "Unknown"
            return FileItem(name, path, path.endswith('/'), access_level=AccessLevel.NO_ACCESS, access_error=ae.message)
        except asyncio.TimeoutError:
            name = path.strip('/').split('/')[-1] if path.strip('/') else "Unknown"
            return FileItem(name, path, path.endswith('/'), access_level=AccessLevel.UNKNOWN, access_error="Timeout retrieving information.")
        except Exception as e:
            if self.debug: print(f"Unhandled error in _get_item_info_sync for {path} ({bucket_name}): {e}")
            name = path.strip('/').split('/')[-1] if path.strip('/') else "Unknown"
            return FileItem(name, path, path.endswith('/'), access_level=AccessLevel.UNKNOWN, access_error=f"Failed to get info: An unexpected error occurred.")


    def _build_breadcrumbs(self, current_s3_path: str, bucket: str, default_prefix: str) -> List[Any]:
        items = []
        bucket_root_s3_path = f"{default_prefix}{bucket}/"
        items.append(dmc.Anchor(f"Bucket: {bucket}", href="#", id={"type": "breadcrumb-item", "path": bucket_root_s3_path, "bucket": bucket}))

        if not current_s3_path.startswith(bucket_root_s3_path) or current_s3_path == bucket_root_s3_path:
            return items

        relative_key = current_s3_path[len(bucket_root_s3_path):].strip('/')
        if not relative_key: return items
            
        parts = [p for p in relative_key.split('/') if p]
        accumulated_key_part = ""
        for part_name in parts:
            accumulated_key_part += part_name + "/"
            breadcrumb_path = f"{bucket_root_s3_path}{accumulated_key_part}"
            items.append(dmc.Anchor(part_name, href="#", id={"type": "breadcrumb-item", "path": breadcrumb_path, "bucket": bucket}))
        return items

    def _create_column_ui(self, items: List[FileItem], col_idx: int, col_s3_path: str, bucket: str, current_nav_path: str) -> dmc.Paper:
        column_items_children = []
        connector = self.connectors[bucket]
        # Check if the column path itself (the path being listed) had an access error during list_items
        # This is implicitly handled if _get_items_sync raises AccessError, which is caught by update_browser_view.
        # If _get_items_sync returns an empty list due to an error that wasn't an AccessError, this won't show.
        # For direct column access errors (e.g. trying to list a restricted prefix), update_browser_view's AccessError catch handles it.

        if not items and not connector.get_access_error(col_s3_path): # If empty and no explicit access error for the column path
            column_items_children.append(dmc.Text("Empty", c="dimmed", ta="center", p="md"))
        elif connector.get_access_error(col_s3_path): # If there was an access error for this column's path directly
             column_items_children.append(dmc.Alert(f"Cannot list: {connector.get_access_error(col_s3_path)}", title="Column Access Error", color="red", m="xs", withCloseButton=True))

        for item in items:
            is_active_in_path = item.is_directory and current_nav_path.startswith(item.path)
            bg_color = "var(--mantine-color-blue-light)" if is_active_in_path else "transparent"
            
            icon, icon_color, text_color = ("tabler:file", "gray", "inherit")
            if item.is_directory: icon, icon_color = ("tabler:folder", "blue")
            
            if item.access_level == AccessLevel.NO_ACCESS:
                icon = "tabler:folder-off" if item.is_directory else "tabler:file-off"
                icon_color, text_color = "red", "red"
            elif item.access_level == AccessLevel.READ_ONLY and item.is_directory:
                icon_color = "orange"
            
            item_tooltip_text = f"""{self._format_size(item.size) + ' - ' if not item.is_directory and item.size is not None else ''}{item.access_level.value}{' - ' + item.last_modified.strftime('%Y-%m-%d %H:%M') if item.last_modified else ''}"""
            item_tooltip_text_full = f"""
            Name: {item.name}
            - Access Level: {item.access_level.value}
            - Size: {self._format_size(item.size) if item.size is not None else 'N/A'}
            - Last Modified: {item.last_modified.strftime('%Y-%m-%d %H:%M') if item.last_modified else 'N/A'}
            """
            item_button = dmc.UnstyledButton(
                id={"type": "browser-item", "identity": item.path, "bucket": bucket},
                children=dmc.Group([
                    DashIconify(icon=icon, color=icon_color, width=40),
                    dmc.Stack([
                        dmc.Text(item.name, size="sm", c=text_color, truncate="end", maw=200),
                        dmc.Text(),
                        dmc.Badge(
                                item_tooltip_text,
                                variant="light",
                                color="#339af0",
                                size="md",
                                radius="xl",
                            ),
                    ], gap=0)
                ], gap="sm", wrap="nowrap"),
                w="100%", p="xs", m="1px",
                styles={"textAlign": "left", "borderRadius": "var(--mantine-radius-sm)", "backgroundColor": bg_color, "&:hover": {"backgroundColor": "var(--mantine-color-action-hover)"}}
            )
            column_items_children.append(dmc.Tooltip(label=item_tooltip_text_full, position="bottom-start", withArrow=True, multiline=True, w=220, children=[item_button]))
        
        column_content_stack = dmc.Stack(column_items_children, gap=1)
        column_scrollable_content = dmc.ScrollArea(children=column_content_stack, h="100%", type="auto", style={"overflowY": "auto", "overflowX": "hidden"})

        return dmc.Paper(column_scrollable_content, p="xs", w=300, miw=280, h="100%", radius="sm", shadow="sm", withBorder=False,)

    def _format_size(self, size_bytes: Optional[int]) -> str:
        if size_bytes is None: return ""
        if size_bytes == 0: return "0 B"
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']; i = 0; num = float(size_bytes)
        while num >= 1024 and i < len(units) - 1: num /= 1024.0; i += 1
        return f"{num:.1f} {units[i]}"
    
    def run(self, host="127.0.0.1", port=8050, debug=True):
        if not self.bucket_connector_pairs:
             print("\nERROR: No buckets configured. S3 Browser may not be functional.\nProvide bucket_connector_pairs to the S3Browser constructor.\nExample: S3Browser([('my-bucket', MyS3Connector())])\n")
        print(f"S3 Browser starting on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

