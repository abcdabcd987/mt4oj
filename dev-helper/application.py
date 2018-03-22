#!/usr/bin/env python3

import dev_helper
import dev_helper.views
from dev_helper import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8322, debug=True)
