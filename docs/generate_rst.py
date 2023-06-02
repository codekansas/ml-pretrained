"""Script which auto-generates RST files for the API reference."""

import os

source_dir = "../pretrained"


def create_rst(module_name: str) -> None:
    content = f"""
{module_name}
{'=' * len(module_name)}

.. automodule:: {module_name}
   :members:
   :undoc-members:
   :show-inheritance:
"""
    with open(f"{module_name}.rst", "w") as f:
        f.write(content.strip())


# Walk the source directory and create RST files for all Python modules
all_files = []
for root, _, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".py") and file != "__init__.py":
            module_path = os.path.join(root, file)
            module_name = os.path.splitext(module_path.replace(os.path.sep, "."))[0].lstrip(".")
            create_rst(module_name)
            all_files.append(module_name)

# Create a file for the current subdirectory
index_content = """
ml-pretrained
=============

Welcome to the documentation for the `ml-pretrained` package!

This is the documentation for the project `here <https://github.com/codekansas/ml-pretrained>`_.

Install these pretrained models using the following command:

.. code-block:: bash

   pip install ml-pretrained

.. toctree::
   :maxdepth: 3
   :caption: API Reference
   :hidden:

"""
for file in sorted(all_files):
    index_content += f"   {file}\n"

with open("index.rst", "w") as f:
    f.write(index_content.strip())
