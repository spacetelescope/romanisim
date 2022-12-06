import os
import urllib.request
import tempfile
import tarfile
import pkg_resources


cosmos_url = ('https://github.com/GalSim-developers/GalSim/raw/releases/'
              '2.4/examples/data/real_galaxy_catalog_23.5_example.fits')
webbpsf_url = ('https://stsci.box.com/shared/static/'
               '963l3m4hcrpc29bqxq68ilcsfgfqwiyc.gz')
tempdir = None


def _download_url(url, filename):
    req = urllib.request.Request(url)
    # marking the below as nosec; we're reading a fixed URL
    with urllib.request.urlopen(req) as r, open(filename, 'wb') as f:  # nosec
        f.write(r.read())


def pytest_configure(config):
    global tempdir
    tempdir = tempfile.TemporaryDirectory()
    if os.environ.get('WEBBPSF_PATH', None) is None:
        os.environ['WEBBPSF_PATH'] = os.path.join(tempdir.name, 'webbpsf-data')
        dirname = pkg_resources.resource_filename('romanisim', 'data')
        outfn = os.path.join(dirname, 'minimal-webbpsf-data.tar.gz')
        tf = tarfile.open(outfn, mode='r:gz')
        tf.extractall(path=tempdir.name)
    if os.environ.get('GALSIM_CAT_PATH', None) is None:
        baseurl = cosmos_url.split('/')[-1]
        os.environ['GALSIM_CAT_PATH'] = os.path.join(tempdir.name, baseurl)
        for rr in ['.fits', '_selection.fits', '_fits.fits']:
            _download_url(
                cosmos_url.replace('.fits', rr),
                os.path.join(tempdir.name, baseurl.replace('.fits', rr)))


def pytest_unconfigure(config):
    global tempdir
    if tempdir is not None:
        tempdir.cleanup()
