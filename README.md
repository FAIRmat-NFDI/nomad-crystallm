# nomad-crystallm

A NOMAD plugin for generating candidate crystal structures from chemical
compositions with [CrystaLLM](https://crystallm.com/) and saving them as NOMAD entries. Run inference from the action
form and save generated CIFs and result entries in your project.

**Use [CrystaLLM on NOMAD Oasis](https://nomad-lab.eu/oasis).**

## Installation

Install the plugin with its inference-worker dependencies:

```sh
pip install 'nomad-crystallm[cpu-action]'
```

Running inference requires a configured NOMAD deployment and a CPU action worker.
Users of an existing Oasis can go straight to the action form.

## Documentation

See the [documentation](https://fairmat-nfdi.github.io/nomad-crystallm/):

- [Using the action form](https://fairmat-nfdi.github.io/nomad-crystallm/how_to/use_this_plugin.html): fill in prompts, choose settings, run inference, and find results.
- [Installation and Oasis setup](https://fairmat-nfdi.github.io/nomad-crystallm/how_to/install_this_plugin.html): add the plugin to an existing Oasis or a local NOMAD installation.

## Contributing

See [Contribute to this plugin](https://fairmat-nfdi.github.io/nomad-crystallm/how_to/contribute_to_this_plugin.html)
for development setup, tests, linting, and documentation previews.

## License

Licensed under the MIT License. See [LICENSE](LICENSE).
