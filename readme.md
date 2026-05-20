# Execution Instructions

## Python version 3.12

To create a virtual environment and install requirements in Python 3.12 on different operating systems, follow the instructions below:

### For Windows:

Open the Command Prompt by pressing `Win + R`, typing `cmd`, and pressing `Enter`.

Change the directory to the desired location for your project:

```sh
cd C:\path\to\project
```

Create a new virtual environment using the `venv` module:

```sh
python -m venv myenv
```

Activate the virtual environment:

```sh
myenv\Scripts\activate
```

Install the project requirements using pip:

```sh
pip install -r requirements.txt
```

### For Linux/Mac:

Open a terminal.

Change the directory to the desired location for your project:

```sh
cd /path/to/project
```

Create a new virtual environment using the `venv` module:

```sh
python3.12 -m venv myenv
```

Activate the virtual environment:

```sh
source myenv/bin/activate
```

Install the project requirements using pip:

```sh
pip install -r requirements.txt
```

These instructions assume you have Python 3.12 installed and added to your system's `PATH` variable.

## Execution Instructions if Multiple Python Versions Installed

If you have multiple Python versions installed on your system, you can use the Python Launcher to create a virtual environment with Python 3.12. Specify the version using the `-p` or `--python` flag. Follow the instructions below:

### For Windows:

Open the Command Prompt by pressing `Win + R`, typing `cmd`, and pressing `Enter`.

Change the directory to the desired location for your project:

```sh
cd C:\path\to\project
```

Create a new virtual environment using the Python Launcher:

```sh
py -3.12 -m venv myenv
```

> **Note**: Replace `myenv` with your desired virtual environment name.

Activate the virtual environment:

```sh
myenv\Scripts\activate
```

Install the project requirements using pip:

```sh
pip install -r requirements.txt
```

### For Linux/Mac:

Open a terminal.

Change the directory to the desired location for your project:

```sh
cd /path/to/project
```

Create a new virtual environment using the Python Launcher:

```sh
python3.12 -m venv myenv
```

> **Note**: Replace `myenv` with your desired virtual environment name.

Activate the virtual environment:

```sh
source myenv/bin/activate
```

Install the project requirements using pip:

```sh
pip install -r requirements.txt
```

By specifying the version using `py -3.12` or `python3.12`, you can ensure that the virtual environment is created using Python 3.12 specifically, even if you have other Python versions installed.
