# Install and configure the plugin

This page is for Oasis administrators and local installations. If CrystaLLM
is already available on your deployment, follow [Using the action form](use_this_plugin.md).

## Installation

Use the Python environment or custom container image of your NOMAD deployment.
The package requires `nomad-lab>=2.0.0` and a NOMAD UI with action forms. Use a
Python version supported by your NOMAD installation.

Install the plugin with its inference dependencies:

```sh
python -m pip install 'nomad-crystallm[cpu-action]'
```

For an installation using the repository version of the plugin:

```sh
python -m pip install 'nomad-crystallm[cpu-action] @ git+https://github.com/FAIRmat-NFDI/nomad-crystallm.git'
```

Use the same plugin version in the API and action-worker environments. Installing
`nomad-crystallm` without the `cpu-action` extra provides the schema dependencies
but does not install everything needed to run inference.

Installing the Python package alone does not start an Oasis or an inference
worker. Complete the setup below before running an action.

## Adding this plugin to NOMAD

Choose the subsection matching your installation:

1. [An existing NOMAD Oasis](#adding-this-plugin-in-your-nomad-oasis).
2. [A local NOMAD installation](#adding-this-plugin-in-your-local-nomad-installation-and-the-source-code-of-nomad).

### Adding this plugin in your NOMAD Oasis

1. Add `nomad-crystallm[cpu-action]` to your deployment's package requirements or
   custom NOMAD image. Rebuild and redeploy the services that use that image;
   installing into a running container alone will not survive its replacement.
2. Make sure your deployment runs Temporal and that NOMAD can connect to it.
   Merge the following into `nomad.yaml`, using your deployment's Temporal host
   and namespace:

   ```yaml
   temporal:
     enabled: true
     host: temporal
     port: 7233
     namespace: default
   ```

   Here, `temporal` is an example container service name. For a native local
   installation, the host may be `localhost`.

3. Run a NOMAD CPU action worker with the plugin installed, the same NOMAD
   configuration as the API, and access to the deployment's upload and action
   storage:

   ```sh
   nomad admin run action-cpu-worker
   ```

   For a container deployment, run this as a worker service using the configured
   NOMAD image and storage mounts. The action uses the CPU task queue; PyTorch
   uses CUDA when it is available to that worker and otherwise runs on CPU.

4. Give the worker access to download the model from Zenodo on its first run.
   Keep its action-artifact storage persistent so later runs can reuse the
   downloaded model.
5. Restart the API and CPU action worker after installing or updating the plugin.
   Sign in to the UI and check that **CrystaLLM Inference** appears under
   **Run action**. If your deployment restricts plugin entry points, allow
   `nomad_crystallm.actions.inference:crystallm_inference` and
   `nomad_crystallm.schemas:crystallm_schemas`.

For deployments with prompt-table upload support, also merge the following into
`nomad.yaml`. This preserves the default Action Assets file types and adds CSV
and XLSX; retain any additional types your deployment already allows.

```yaml
actions:
  action_assets:
    allowed_media_types:
      - audio/*
      - image/*
      - application/pdf
      - application/json
      - text/plain
      - application/octet-stream
      - text/csv
      - application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
```

Restart the API and worker after changing the configuration. See the
[NOMAD Oasis plugin configuration documentation](https://docs.nomad-lab.eu/howto/oasis/configure.html#plugins)
for deployment-specific plugin setup.

### Adding this plugin in your local NOMAD installation and the source code of NOMAD

Install the package in the environment used by your local NOMAD API and CPU
action worker, then apply the Temporal and storage setup above.

For development across NOMAD and the plugin, use
[`nomad-distro-dev`](https://github.com/FAIRmat-NFDI/nomad-distro-dev). See
[Developing on `nomad-distro-dev`](contribute_to_this_plugin.md#developing-on-nomad-distro-dev) below for the
workspace setup and service commands.
