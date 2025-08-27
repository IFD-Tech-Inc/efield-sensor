#!/usr/bin/env python3
"""
Entry point for IFD Signal Analysis Utility.

This is a backward compatibility wrapper that allows users to run the application
with `python main.py` instead of requiring `python -m ifd_signal_analysis`.
"""

if __name__ == '__main__':
    from ifd_signal_analysis.__main__ import main
    main()
