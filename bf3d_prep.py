import os
import ipywidgets as widgets
from IPython.display import display, clear_output

class FolderSelector:
    def __init__(self, rootfolder, filetype, return_type = 'path'):
        self.rootfolder = rootfolder
        self.filetype = filetype
        self.current_path = self.rootfolder
        self.display_options(self.current_path)
        self.selected_file = None
        self.return_type = return_type

    def display_options(self, folder_path):
        clear_output(wait=True)  # Clear the previous widgets
        options = self.get_options(folder_path)
        if not options:
            print("No folders or files matching the criteria.")
            return

        dropdown = widgets.Dropdown(
            options=options,
            description='Select:',
            disabled=False,
        )

        button = widgets.Button(description="Select")
        output = widgets.Output()

        display(dropdown, button, output)

        def on_button_clicked(b):
            with output:
                #if the files in the folder is end with the file type, then return the folder path
                if self.return_type == 'path':
                    if dropdown.value.endswith(self.filetype):
                        self.selected_file = os.path.join(folder_path, dropdown.value)
                        print(f"File selected: {self.selected_file}")
                    else:
                        self.current_path = os.path.join(folder_path, dropdown.value)
                        self.display_options(self.current_path)

        button.on_click(on_button_clicked)

    def get_options(self, folder_path):
        options = ['..']  # Option to go up one directory
        try:
            for item in os.listdir(folder_path):
                full_path = os.path.join(folder_path, item)
                if os.path.isdir(full_path) and not item.startswith('.'):
                    options.append(item)
                elif item.endswith(self.filetype) and os.path.isfile(full_path):
                    options.append(item)
        except Exception as e:
            print(f"Error accessing the directory: {e}")
            options = []

        return options
    
    def return_folder(self):
        """automatic scanning the files in the folder
        if the files in the folder is end with the file type, then return the folder path"""
        

# Example usage
# folder_selector = FolderSelector('/path/to/rootfolder', '.txt')
