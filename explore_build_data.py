import os
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa:E402
import preprocessor as pre


plt.figure(9870)
plt.clf()
ax = plt.gca()

body_size, build, breed = pre.calculate_build_data()
ax.plot(body_size, build, '.')
ax.set_xlabel("Body size")
ax.set_ylabel("Build (positive=stocky, negative=lanky)")

title="Dog breeds by size and build"
ax.set_title(title)
title_str = "_".join(title.lower().split())
output_filename = title_str + ".png"
plt.savefig(output_filename)
