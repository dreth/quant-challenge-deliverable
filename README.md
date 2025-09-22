# quant-challenge-deliverable

This repo contains all the code written as a self contained project.

I used `uv` and python 3.12.11 to setup this project and run the code.

Overall I had this separated in multiple scripts, but the challenge calls for a single script, so I have combined everything in a single script `main.py`.

To setup the project with its own venv (if you're not familiar with `uv`), first install `uv`, and then:

```bash
uv venv .venv --python 3.12.11
```

Then sync the dependencies:

```bash
uv sync
```

To run the code, make sure to activate the venv first:

```bash
source .venv/bin/activate
```

And then run the script:

```bash
python main.py
```

The script automatically generates directories for the reports, data and artifacts, but I have intentionally added these to the .gitignore so that they are not tracked by git.
