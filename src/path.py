# Small helper file to get the export command (Python 3+)
import os, warnings

def return_export(path="/", v=False):
    """
    Return or print the proper export path to execute.
    path takes a string ["/"] to where voxelmorph is cloned to
    v is [False] or True to print in addition to returning the export command
    """
    if not os.path.exists(f'{path}/ext/neuron/'):
        warnings.warn(f'{path}/ext/neuron/ does not exist.', Warning)
    if not os.path.exists(f'{path}/ext/pynd-lib/'):
        warnings.warn(f'{path}/ext/pynd-lib/ does not exist.', Warning)
    if not os.path.exists(f'{path}/ext/pytools-lib/'):
        warnings.warn(f'{path}/ext/pytools-lib/ does not exist.', Warning)
    
    export = f"export PYTHONPATH=$PYTHONPATH:{path}/ext/neuron/:{path}/ext/pynd-lib/:{path}/ext/pytools-lib/"
    
    if v:
        print(export)
    return export

if __name__ == "__main__":
    
    path = input("Enter Path to Voxelmorph: ")
    return_export(path, v=True)
