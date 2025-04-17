"""
Core subpackage of the ACIP method providing all required code to build an `ACIPModel`.
This package is fully self-contained and does not depend on any other subpackages in `acip`.
It only uses relative imports to enable seamless saving and loading with the Huggingface transformers library
(`from_pretrained`, `save_pretrained`, `push_to_hub`), including remote code.
"""
