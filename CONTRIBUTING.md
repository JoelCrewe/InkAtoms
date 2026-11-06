# Contributing

Bug reports, examples and pull requests are welcome.

```bash
git clone https://github.com/JoelCrewe/InkAtoms.git
cd InkAtoms
python -m pip install -e ".[dev,docs]"
python -m pytest --mpl          # tests, including image comparison
python -m sphinx -b html docs docs/_build/html
```

If a change alters the drawings on purpose, regenerate the image baselines with
`pytest --mpl-generate-path=tests/baseline tests/test_images.py` and say so in the pull request.

Please keep new features drawing through the depth-sorted scene in `render.py`, so
everything overlaps correctly, and add a test and (for visible features) a gallery example.
