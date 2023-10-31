#!/usr/bin/env zsh
# Warning: WIP, use at own risk.
# Usage example: lmland.sh pr80

sl goto main && sl pull && sl merge "$1";
sl push -r "$1" --to main && sl push -d "$1"