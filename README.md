# UV
To install certain version of Python.
```Bash
uv python install [版本号]
```
To init a new project.
```Bash
uv init [项目名]
```
Or init a project right in current dir.
```Bash
uv init
```
To install Python for current project.
```Bash
uv venv --python [版本号]
```
Or use a specific Python version in current dir.
```Bash
uv python pin [版本号]
```
To run Python scripts.
```Bash
uv run [*.py]
```
To pip install packages, which will automatically record dependencies in pyproject.toml file.
```Bash
uv add [package name]
```
