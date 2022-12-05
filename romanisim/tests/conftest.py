import os
import urllib
import tempfile
import tarfile


cosmos_url = ('https://github.com/GalSim-developers/GalSim/raw/releases/'
              '2.4/examples/data/real_galaxy_catalog_23.5_example.fits')
webbpsf_url = ('https://stsci.box.com/shared/static/'
               '963l3m4hcrpc29bqxq68ilcsfgfqwiyc.gz')

print('hello!')


def _download_url(url, filename):
    req = urllib.request.Request(url)
    # marking the below as nosec; we're reading a fixed URL
    with urllib.request.urlopen(req) as r, open(filename, 'wb') as f:  # nosec
        f.write(r.read())


def pytest_configure(config):
    tempdir = tempfile.TemporaryDirectory()
    config.romanisim_test_tempdir = tempdir
    if os.environ.get('WEBBPSF_PATH', None) is None:
        os.environ['WEBBPSF_PATH'] = tempdir.name
        outfn = os.path.join(tempdir.name, 'minimal-webbpsf-data.tar.gz')
        _download_url(webbpsf_url, outfn)
        tf = tarfile.Tarfile(outfn)
        tf.extractall(path=tempdir.name)
    if os.environ.get('GALSIM_CAT_PATH') is None:
        os.environ['GALSIM_CAT_PATH'] = tempdir.name
        baseurl = cosmos_url.split('/')[-1]
        for rr in ['.fits', '_selection.fits', '_fits.fits']:
            _download_url(
                cosmos_url.replace('.fits', rr),
                os.path.join(tempdir.name, baseurl.replace('.fits', 'rr')))


def pytest_unconfigure(config):
    tempdir = getattr(config, 'romanisim_test_tempdir', None)
    if tempdir is not None:
        tempdir.cleanup()
