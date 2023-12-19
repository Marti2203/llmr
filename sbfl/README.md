# sbfl
A simple SBFL implementation for fix-localization

## Usage
Make sure to initialize submodules.

```
  ./instrument.sh <absolute path to binary>
```

Will output `<basename>.tracer`.

Running the program will print the trace to `stderr`.
The `TRACE_FILE` environment variable can be used to otherwise direct the trace to a file.

Once you have a trace, hopefully `addr2line` can decode it.
