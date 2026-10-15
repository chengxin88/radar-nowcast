"""Optional cloud nowcast entry; shared locking, dates and SQLite state."""
import sys
import cloud_runner

if __name__ == '__main__':
    sys.argv.insert(1, 'nowcast')
    sys.exit(cloud_runner.main())
