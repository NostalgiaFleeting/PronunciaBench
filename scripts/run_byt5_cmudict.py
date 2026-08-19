"""Compatibility entry point for the ByT5 CMUdict experiment."""

if __package__:
    from .byt5_experiment import *  # noqa: F403
    from .byt5_experiment import main
else:
    from byt5_experiment import *  # noqa: F403
    from byt5_experiment import main

if __name__ == "__main__":
    main()
