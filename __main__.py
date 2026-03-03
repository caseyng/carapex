"""
__main__.py
-----------
CLI entrypoint for carapex.

Usage:
    python -m carapex --init              # write config.json to current dir
    python -m carapex --show-config       # print all config with defaults
    python -m carapex --check             # health check against config.json
    python -m carapex --check --config x  # health check against x.json
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog        = "python -m carapex",
        description = "carapex — hardened prompt execution boundary",
    )
    parser.add_argument(
        "--init",
        action  = "store_true",
        help    = "Write default config.json to current directory",
    )
    parser.add_argument(
        "--show-config",
        action  = "store_true",
        help    = "Print all config keys with types and defaults",
    )
    parser.add_argument(
        "--check",
        action  = "store_true",
        help    = "Health check backends defined in config",
    )
    parser.add_argument(
        "--config",
        default = "config.json",
        help    = "Path to config file (default: config.json)",
    )

    args = parser.parse_args()

    if args.init:
        from . import config
        config.write_default("config.json")
        print("Edit config.json, then run: python -m carapex --check")
        sys.exit(0)

    if args.show_config:
        from . import config
        config.explain()
        sys.exit(0)

    if args.check:
        import os
        from . import build, config
        from .exceptions import (
            BackendUnavailableError,
            ConfigurationError,
            PluginNotFoundError,
        )

        if not os.path.exists(args.config):
            print(f"Config file not found: {args.config}")
            print("Run: python -m carapex --init")
            sys.exit(1)

        try:
            cp = build(args.config)
            cp.health_check()
            print(f"✓ Backend ready — {cp!r}")
            cp.close()
            sys.exit(0)
        except BackendUnavailableError as e:
            print(f"✗ {e}")
            sys.exit(1)
        except (ConfigurationError, PluginNotFoundError, ImportError,
                FileNotFoundError, ValueError) as e:
            print(f"✗ Build failed: {e}")
            sys.exit(1)

    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
