PLaMo Translation Model Test
============================

A tiny Python package to test PLaMo Translation Model with MPS
on Apple Silicon or CPU in general.

Usage
-----

Clone this repository and run following commands or equivalent ones.

```
$ python3 -m venv .venv
$ .venv/bin/pip3 install -e .
$ .venv/bin/plamo_2_translate_test translate 'Hello, World!'
```

Initial run would download 20GB or more model data and loading model,
and every run would load model which takes a few seconds.

It can also run HTTP server and keep it in memory, then translate with `--server`.

```
$ .venv/bin/plamo_2_translate_test server
$ .venv/bin/plamo_2_translate_test translate --server 'Hello, World!'
```

With `--stream`, it streams trenslation.

```
$ .venv/bin/plamo_2_translate_test translate --server --stream 'Hello, World!'
```

License
-------

See `LICENSE` file in each dependency modules.

Also see <https://huggingface.co/pfnet/plamo-2-translate> for
the model license.
