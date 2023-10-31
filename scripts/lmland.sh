#!/usr/bin/env zsh

# WIP
sl goto main && sl pull && sl merge "$1" && sl push -r "$1" --to main && sl push -d "$1"