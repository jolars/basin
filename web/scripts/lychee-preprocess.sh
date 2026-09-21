#!/usr/bin/env sh
set -eu

# Check mdsvex's external links without changing the HTML we publish.
sed 's/ rel="nofollow"//g' "$1"
