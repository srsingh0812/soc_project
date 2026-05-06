import os

# List of folders to create
folders = [
    'data/raw/panasonic_18650pf',
    'data/processed',
    'notebooks',
    'src/data',
    'src/models',
    'src/training',
    'src/evaluation',
    'src/utils',
    'api',
    'tests',
    'models_saved',
    'scripts',
    'docs'
]

# List of directories where __init__.py should be created
init_dirs = [
    'src',
    'src/data',
    'src/models',
    'src/training',
    'src/evaluation',
    'src/utils',
    'tests'
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

# Create empty __init__.py files
for init_dir in init_dirs:
    init_file = os.path.join(init_dir, '__init__.py')
    with open(init_file, 'w') as f:
        pass  # Create empty file
    print(f"Created __init__.py in: {init_dir}")

print("Project structure setup complete!")