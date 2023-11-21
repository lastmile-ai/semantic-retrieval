# Summary

- Build a custom CLI chat bot using AIConfig. Primary use case is interactive code modification, but can be used as a general-purpose chat bot.
- Use simple AIConfig runtime API to run an arbitrary-length sequence of prompts.
- Leverage streaming callbacks to allow gracefully interrupting LLM output and going back to the query prompt.

# Usage

## Quick start:

`conda env create -n climate --file python/src/scripts/Cli-Mate/climate-conda.yml && conda activate climate && python python/src/scripts/Cli-Mate/cli-mate.py -c python/src/scripts/Cli-Mate/cli-mate.aiconfig.json loop`

Enter 'h', 'help', or '?' for help.
> help

Example:

```
Enter 'h', 'help', or '?' for help.
> help

Exit loop: Ctrl-D
Toggle multiline input mode: m or multiline
Clear screen: c or clear
Reload source code: r or reload

> capital of nys
Albany
> what was my prev question about?
Your previous question was about the capital of New York State (NYS).
> m
Multiline input mode: on
Hit option-enter to submit.
> the
  quick
  brown

Your fragment appears to be the start of a well-known phrase, "The quick brown fox." The full sentence is typically "The quick brown fox jumps over the lazy dog."
>
```


## With source code:

contents of python/test.py: `import open`

```
% python cookbooks/Cli-Mate/cli-mate.py -c cookbooks/Cli-Mate/cli-mate.aiconfig.json loop -scf='python/test.py '
Query: [ctrl-D to exit] summarize the code.
The given code attempts to import a module named 'open' in Python. But it seems incorrect as 'open' is a built-in function in Python to open files but not a module. This code will throw an error if executed.
# fix the code in python/test.py to `import os` while cli-mate is still running.

Query: [ctrl-D to exit] reload what does the code do now?
The code imports the module 'os' in Python.
```

Also see: `python cookbooks/Cli-Mate/cli-mate.py -h`
