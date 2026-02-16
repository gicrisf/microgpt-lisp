# microgpt-lisp

A minimal GPT training and inference engine in a single file, with a custom autograd. This is a Common Lisp re-implementation of Andrej Karpathy's [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) that I wrote for learning reasons.

## Running

```bash
sbcl --load microgpt.asd --eval '(asdf:load-system :microgpt)' --eval '(microgpt:main)' --quit
```

Or from Sly/SLIME:

```lisp
(asdf:load-system :microgpt)
(microgpt:main)
```

## Requirements

- SBCL (or any ASDF-compatible CL implementation)
- An `input.txt` file with one name per line
