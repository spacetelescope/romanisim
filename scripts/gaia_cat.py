#!/usr/bin/env python -u

"""
Gaia catalog generator for roman simulator.

"""

import argparse
import astroquery
from astroquery.gaia import Gaia
from astropy.time import Time
from romanisim import gaia
import romanisim.parameters, romanisim.bandpass

def main():
    """Main function of error estimator"""

    # Define the parser
    parser = argparse.ArgumentParser(description='Command and options for estimating fit uncertainties')

    # Command line Argument
    parser.add_argument('radecrad',
                        type=float,
                        metavar='178.389741 -2.678655, 1.0',
                        nargs=3,
                        help=f"The right ascension, declination, and radius of Gaia objects to catalog."
                        )

    # Optional output filename
    parser.add_argument('-o', '--output',
                        type=str,
                        metavar='gaia-178-2-2027-06-01',
                        help=f"Name prefix of gaia catalog file."
                        )

    # Optional Time
    parser.add_argument('-t', '--time',
                        type=str,
                        metavar='2027-06-01',
                        default='2027-06-01',
                        help=f"Time of observations of sources in catalog."
                        )

    # Debug argument
    parser.add_argument('-d', "--debug",
                        action='store_true',
                        help='Display verbose debug output')

    # Collect the arguments
    args = parser.parse_args()

    if args.debug:
        print("args = " + str(args))
    
    q = f'select * from gaiadr3.gaia_source where distance({args.radecrad[0]}, {args.radecrad[1]}, ra, dec) < {args.radecrad[2]}'
    job = Gaia.launch_job_async(q)
    r = job.get_results()
    len(r)
    outfile_name = f'gaia_{args.radecrad[0]:.2f}_{args.radecrad[1]:.2f}_{args.radecrad[2]:.2f}-{args.time}.ecsv'
    if args.output:
        outfile_name = args.output + ".ecsv"
    gaia.gaia2romanisimcat(r, Time(args.time), fluxfields=set(romanisim.bandpass.galsim2roman_bandpass.values())).write(outfile_name, overwrite=True)

# Call main if run (as opposed to imported as a module)
if __name__ == "__main__":
    main()
