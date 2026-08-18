import os
import sys
import zipfile


def package_project():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_name = os.path.basename(project_root)
    parent_dir = os.path.dirname(project_root)
    zip_path = os.path.join(parent_dir, project_name + '.zip')

    excluded_dirs = {'__pycache__', '.git', '.venv', 'venv', 'node_modules'}
    excluded_exts = {'.pyc', '.pyo', '.egg-info'}

    count = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(project_root):
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            for f in files:
                if any(f.endswith(ext) for ext in excluded_exts):
                    continue
                full_path = os.path.join(root, f)
                arc_name = os.path.relpath(full_path, project_root)
                zf.write(full_path, arc_name)
                count += 1

    print("Project packaged: {}".format(zip_path))
    print("Files included: {}".format(count))
    return zip_path


if __name__ == '__main__':
    package_project()
