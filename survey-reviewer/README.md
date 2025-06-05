# Review app

The purpose of this app is to plot all the images from a flight, so we can just eyeball them for anything funny e.g.

- Check camera orientation
- Remove test images taken before the survey actually started.
- Look for some cameras significantly darker than others etc.

## Running

Run the `prepare_data.ipynb` notebook to generate data - this takes a while! Then run `python server.py` and `yarn dev`.