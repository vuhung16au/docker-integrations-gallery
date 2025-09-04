#!/usr/bin/env python3
"""
Simple Hello World application for Docker demonstration with non-root user
"""

import getpass

def main():
    print("Hello, World! (non-root user)")
    print(f"Current user: {getpass.getuser()}")

if __name__ == "__main__":
    main()
