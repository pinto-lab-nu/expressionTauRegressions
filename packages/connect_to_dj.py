# """ General purpose module to connect to DataJoint database, create virtual 
# modules, and set up external storage.
# """

# import datajoint as dj
# import platform

# VM = None # Global variable to store virtual modules

# def my_os():
#   my_os = platform.system()

#   if   my_os == 'Windows':
#     return 'PC'
  
#   elif my_os == 'Darwin':
#     return 'Mac'
  
#   elif my_os == 'Linux':
#     # Need to figure out whether this is quest or workstation
#     if platform.node()[0]=='q':
#       return 'Quest'
    
#     else:
#       return 'Linux'
    
#   else:
#     print('Warning: unknown OS')
#     return None


# def connection_info():
#     # DataJoint version details:
#     print("DataJoint version:")
#     print(dj.__version__)

#     # Should say: DataJoint connection (connected)... If not, check the LogIn configuration
#     print(dj.conn())


# def _get_virtual_modules(modules=None, verbose=False):

#     # Get schemas in database
#     schemas = dj.list_schemas()
#     prefix = "pintolab_"  # Prefix used for *most* schemas

#     if modules == None:
#         modules = schemas
#     elif not isinstance(modules, list):
#         modules = [modules]
    
#     # Check if virtual modules have already been created
#     global VM
#     if VM is None:
#         VM = {}
#     for module in modules:
#         if module in VM or module.split("_")[-1] in VM:
#             continue

#         if prefix + module in schemas:
#             VM[module] = dj.create_virtual_module(module, prefix + module)
#             if verbose and module in schemas:
#                 print(
#                     'Warning : both schema "'
#                     + module
#                     + '" and schema "'
#                     + prefix
#                     + module
#                     + '" exist in the database.'
#                     + " Priority is given to the latter."
#                 )

#         elif module.startswith(prefix) and module in schemas:
#             VM[module[len(prefix) :]] = dj.create_virtual_module(module, module)

#         elif module in schemas:
#             # Currently ignoring schemas that don't start with prefix
#             pass
#         else:
#             if verbose:
#                 print("Warning: module " + module + " not found in schema list.")

#         # Temporary workaround for misnamed database schema
#         if module == "widefield_redux":
#             VM[module] = dj.create_virtual_module(module, "pintolab")
#         elif module == "widefield_trial_df":
#             VM[module] = dj.create_virtual_module(module, module)
#             if verbose:
#                 print(
#                     f"Warning: using {module} instead of {prefix + module} for this special case."
#                 )

#     if verbose:
#         print("List of virtual modules:")
#         print(list(VM.keys()))

#     return VM


# def get_virtual_modules(modules=None, verbose=False):
#     global VM
#     assert VM is not None, "Virtual modules should be created on import"
#     if verbose:
#         print("List of virtual modules:")
#         print(list(VM.keys()))
#     return VM


# def config_connection(db_id=2):
#     # LogIn configuration:
#     if db_id == 1:
#         # Original database host
#         dj.config["database.host"] = "vfsmphysiomdb01.fsm.northwestern.edu"
#     elif db_id == 2:
#         # Database host after 2025 migration
#         dj.config["database.host"] = "vfsmphysiomdb2.fsm.northwestern.edu"
#     dj.config["database.user"] = "ward5243user"
#     dj.config["database.password"] = "mouseVR&Ca2+"


# def config_external_storage(verbose=True):

#     VM = get_virtual_modules("lab", verbose=False)

#     # Set up external storage
#     os_name = my_os()

#     pathQ = VM["lab"].Path() & 'path_type = "dj_external"' & f'os = "{os_name}"'
#     path = pathQ.fetch("server_path")[0]

#     storage = {"protocol": "file", "location": path}
#     dj.config["stores"] = {"extstorage": storage}

#     if verbose:
#         print(f"Set up external storage at {path}")


# # Connect to database
# config_connection()
# dj.conn()

# # Get virtual modules
# _get_virtual_modules()

# # Set up external storage
# config_external_storage()
