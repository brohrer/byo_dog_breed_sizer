import sys
import numpy as np
import data_handler as dta
import dog_sizer as siz


breed, height, mass = dta.get_data()

if len(sys.argv) < 2 or sys.argv[1] not in breed:
    print()
    print("Usage:")
    print("    > python3 sizer_command_line.py \"valid breed\"")
    print()
    print("Some breeds you can try:")
    print("   ", ", ".join(np.random.choice(breed, size=5)))
else:
    stocky, medium, lanky = siz.get_similar_breeds(sys.argv[1])
    print("Stocky breeds:", ", ".join(stocky))
    print("Medium breeds:", ", ".join(medium))
    print("Lanky breeds:", ", ".join(lanky))
