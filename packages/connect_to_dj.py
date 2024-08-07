''' General purpose module to connect to DataJoint database, create virtual 
modules, and set up external storage.
'''
import datajoint as dj 
import platform

def connection_info():
  # DataJoint version details: 
  print('DataJoint version:')
  print(dj.__version__)

  # Should say: DataJoint connection (connected)... If not, check the LogIn configuration 
  print(dj.conn()) 

def get_virtual_modules(modules=None,verbose=True):
    
  # Get schemas in database
  schemas = dj.list_schemas()
  prefix = 'pintolab_' # Prefix used for *most* schemas
  

  if modules == None:
    modules = schemas
  elif not isinstance(modules, list):
    modules = [modules]

  VM = {}
  for module in modules:

    if prefix + module in schemas:
      VM[module] = dj.create_virtual_module(module, prefix + module)
      if verbose and module in schemas:
        print('Warning : both schema "' + module + '" and schema "' 
        +  prefix + module + '" exist in the database.' 
        + ' Priority is given to the latter.')

    elif module.startswith(prefix) and module in schemas:
      VM[module[len(prefix):]] = dj.create_virtual_module(module, module)

    elif module in schemas:
      # Currently ignoring schemas that don't start with prefix
      pass
    else:
      if verbose:
        print('Warning: module ' + module + ' not found in schema list.')

    # Temporary workaround for misnamed database schema
    if module == 'widefield_redux':
      VM[module] = dj.create_virtual_module(module,'pintolab')
    elif module == 'widefield_trial_df':
      VM[module] = dj.create_virtual_module(module,module)
      if verbose:
        print(f'Warning: using {module} instead of {prefix + module} for this special case.')
  
  if verbose:
    print('List of virtual modules:')
    print(list(VM.keys()))

  return(VM)

def config_connection():
  # LogIn configuration:
  dj.config['database.host'] = 'vfsmphysiomdb01.fsm.northwestern.edu'
  dj.config['database.user'] = 'ward5243user'
  dj.config['database.password'] = 'mouseVR&Ca2+'

def config_external_storage(verbose=True):
  
  VM = get_virtual_modules('lab',verbose=False)

  # Set up external storage (note: not compatible with quest yet)
  my_os = platform.system()
  if my_os == 'Windows':
    my_os = 'PC'
  elif my_os == 'Darwin':
    my_os = 'Mac'
  pathQ = (VM['lab'].Path() & 'path_type = "root"' & f'os = "{my_os}"')
  server_path = pathQ.fetch('server_path')[0]
  server_path = server_path + '_Dj_external'

  storage = {'protocol': 'file','location': server_path}
  dj.config['stores'] = {'extstorage': storage}

  if verbose:
    print(f'Set up external storage at {server_path}')

# Connect to database
config_connection()
dj.conn()

# Set up external storage
config_external_storage()