# ztfimages

Tool to get ZTF CCD images for both raw and science images

Option to get CCD science image with global CCD flat-field

WARNING
- Use ztfquery
- Require local ZTF data repository with same structure than in ZTF data products rerpository
- Create a new directory /ccdflat/ in your /year/date/ local ZTF data repository
- Add global CCD flat-field images (.fits) in the new /ccdflat/ directory
- Add new quadrant data files in science image directory with same name than ZTF pipeline products but with extension '_in2p3..fits'   
