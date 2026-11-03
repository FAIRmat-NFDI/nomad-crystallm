# Contribute to this plugin

For everyday use, see [Using the action form](use_this_plugin.md). Deployment
administrators can follow [Installation](install_this_plugin.md).

## Development

For contributing to the plugin, clone the repository and install its development
dependencies:

```sh
git clone https://github.com/FAIRmat-NFDI/nomad-crystallm.git
cd nomad-crystallm
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install uv
uv pip install -e '.[dev]'
```

### Developing on `nomad-distro-dev`

Use the dedicated
[`nomad-distro-dev`](https://github.com/FAIRmat-NFDI/nomad-distro-dev) workspace
when developing against local NOMAD packages. Add the plugin to its root
dependencies with the `cpu-action` extra:

```toml
dependencies = [
  # Other workspace dependencies...
  "nomad-crystallm[cpu-action]",
]
```

Follow the workspace README to configure its infrastructure, including Temporal.
Use compatible NOMAD and UI revisions with action-form support. From the workspace
root, run its setup task:

```sh
uv run poe setup
```

Then start these services in separate terminals. Run the API and CPU action
worker commands from the workspace root:

```sh
uv run poe start
```

```sh
uv run poe cpuworker
```

Start the UI with action forms from the workspace's `ui` directory:

```sh
cd ui
npm install
npm start
```

### Run the tests

From the plugin directory:

```sh
python -m pytest -sv tests
```

To generate a coverage report:

```sh
uv pip install pytest-cov
python -m pytest --cov=src tests
```

### Run linting and auto-formatting

```sh
ruff check .
ruff format . --check
```

Run `ruff format .` to apply formatting changes.

### Debugging

Use `python -m pytest --pdb tests` to stop on a failing test. For VS Code, adapt
this configuration in `.vscode/launch.json` to the test you want to inspect:

```json
{
  "configurations": [
    {
      "name": "Debug plugin tests",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/.venv/bin/pytest",
      "args": ["-sv", "--pdb", "tests/test_inference_generation.py"],
      "justMyCode": true
    }
  ]
}
```

### Documentation on Github pages

From the plugin directory:

```sh
uv pip install -r requirements_docs.txt
mkdocs serve
```

## Template update

Plugin maintainers can use [`cruft`](https://cruft.github.io/cruft/) to check for
updates to the project template. Run `cruft update` from the plugin repository.
Oasis administrators and action-form users do not need this step.

## Main contributors

| Name           | E-mail                                                                          |
| -------------- | ------------------------------------------------------------------------------- |
| Ahmed Ilyas    | [ahmed.ilyas@physik.hu-berlin.de](mailto:ahmed.ilyas@physik.hu-berlin.de)       |
| Sarthak Kapoor | [sarthak.kapoor@physik.hu-berlin.de](mailto:sarthak.kapoor@physik.hu-berlin.de) |
