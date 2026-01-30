Build:

```bash
docker build . -t ptoas:py3.12

# optional, to change python version
docker build . -t ptoas:py3.11 --build-arg PY_VER=cp311-cp311
```

Use:

```bash
docker run --rm -it ptoas:py3.12 /bin/bash
```