import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from astropy.io import fits
from astropy.time import Time
import pandas
import os
from ztfquery import query

# Class to get CCD image
class ccdimage:

    def __init__(self, imgfile): 
        self.imgfile = imgfile
    
    def polynom2(self, x, a0, a1, a2):
        # 2nd order polynomial function to fit overscan
        return a0 + a1*x + a2*x**2
    
    def overscan_correction(self, quadrant, overscan):
        # Fit x-overscan pixels (y-stacked in [4:27]) with 2nd-order polynomial function 
        over_profil = np.median(overscan[:,4:27], axis=1)
        xpixel = np.arange(3080)
        over_median = np.median(over_profil)
        df_overscan = pandas.DataFrame({'xpix': xpixel, 'yval': over_profil})
        over_cut = 2*over_median
        flag = df_overscan["yval"]<over_cut
        df_overscan = df_overscan[flag] 
        par0 = np.array([over_median, 0, 0])
        #par, cov = optimize.curve_fit(self.polynom2, xpixel, over_profil, par0)
        par, cov = optimize.curve_fit(self.polynom2, df_overscan["xpix"], df_overscan["yval"], par0)
        err = np.sqrt(np.diag(cov))
        xfit = self.polynom2(xpixel, par[0], par[1], par[2])
        # Make the difference between raw quadrant and x-overscan function
        overfit = xfit[:, np.newaxis]
        return quadrant - overfit

    def read_quadrant(self, img_file, idx):
        if (os.path.isfile(img_file)):
            q = fits.getdata(img_file, idx)
        else:
            raise ValueError("-> Missing file:", img_file)
        return q
    
    def linearity_correction(self, cID, aID, quad_in):
        # Extract linearity coefficients
        ccdID = int(cID)
        ampID = int(aID)-1
        coeff_file = "CCD_amp_coeff_v2.txt"
        coeff = pandas.read_csv(coeff_file, comment='#', header=None, sep='\s+', usecols=[0, 1, 2, 3, 4]) 
        coeff.columns = ['CCDID', 'AMP_NAME', 'AMP_ID', 'A', 'B']
        flag_ccdID = coeff['CCDID']==ccdID
        coeff = coeff[flag_ccdID]
        flag_ampID = coeff['AMP_ID']==ampID
        coeff = coeff[flag_ampID]
        A = float(coeff.loc[coeff.index[0],'A'])
        B = float(coeff.loc[coeff.index[0],'B'])
        # Perform quadratic correction on pixel counting
        quad_out = quad_in / (A*quad_in*quad_in + B*quad_in + 1)
        return quad_out

    def get_ccd_raw(self, img_file=None, overscan_corr=True, linearity_corr=True):
        # Merging of the 4 quadrants (00, 01, 02, 03) of a CCD from raw data
        # in a single image with the quadrant sub-structure
        #             q2 q1
        #             q3 q4
        # Read CCD quadrants
        if (img_file is None):
            img_file = self.imgfile
        q1 = self.read_quadrant(img_file, 1)
        q2 = self.read_quadrant(img_file, 2)
        q3 = self.read_quadrant(img_file, 3)
        q4 = self.read_quadrant(img_file, 4)
        # Read CCD overscans
        if (overscan_corr):
            o1 = self.read_quadrant(img_file, 5)
            o2 = self.read_quadrant(img_file, 6)
            o3 = self.read_quadrant(img_file, 7)
            o4 = self.read_quadrant(img_file, 8)
            # Overscan correction with 2nd-order polynomial fit of overscan
            q1 = self.overscan_correction(q1, o1)
            q2 = self.overscan_correction(q2, o2)
            q3 = self.overscan_correction(q3, o3)
            q4 = self.overscan_correction(q4, o4)    
        # Per-quadrant linearity correction on pixel counting based on a quadratic model fit to laboratory data 
        if (linearity_corr):
            cID = self.get_ccd_id()
            q1 = self.linearity_correction(cID, '1', q1)
            q2 = self.linearity_correction(cID, '2', q2)
            q3 = self.linearity_correction(cID, '3', q3)
            q4 = self.linearity_correction(cID, '4', q4)
        # Each quadrant is rotated by 180° before merging
        q1 = np.rot90(q1, 2)
        q2 = np.rot90(q2, 2)
        q3 = np.rot90(q3, 2)
        q4 = np.rot90(q4, 2)
        # Horizontal merging of CCD quadrants 00 and 01 
        ccd_up = np.concatenate((q2, q1), axis=1) 
        # Horizontal merging of CCD quadrants 02 and 03
        ccd_down = np.concatenate((q3, q4), axis=1) 
        # Vertical merging of the two above half-CCD 
        ccd = np.concatenate((ccd_down, ccd_up), axis=0) 
        return ccd
    
    def image_ploting(self, image, Imin=0, Imax=0):
        # Plot image
        if (Imin==0 and Imax==0):
            m0 = np.median(image)
            s0 = np.std(image)
            Imin = m0 - s0
            Imax = m0 + s0
            if (s0 > m0):
                Imin = 0
                Imax = 2*m0
        fig = plt.figure(figsize=(10, 8))
        fig.add_subplot(111)
        plt.imshow(image, interpolation='nearest', origin='lower', cmap='gray', vmin=Imin, vmax=Imax)
        plt.colorbar()
        #fig.savefig(dirname+filename+".png", dpi=150, bbox_inches='tight')
        return

    def image_saving(self, filename, image):
        # Save image as fit file
        os.system("rm "+filename)
        hdu = fits.PrimaryHDU()
        hdu.writeto(filename)#, clobber=True)
        fits.append(filename, image, overwrite=True)
        #os.system("fpack "+filename)
        #os.system("rm "+filename)   
        return

    def get_ccd_id(self):
        # Extract CCD ID from imgage file name
        idx = self.imgfile.index('.fits') - 4
        return self.imgfile[idx:idx+2]

    def get_filter_id(self):
        # Extract CCD ID from imgage file name
        idx = self.imgfile.index('.fits') - 8
        return self.imgfile[idx:idx+2]

    def get_date(self):
        # Extract date from imgage file name
        idx = self.imgfile.index('/ZTF/raw/') + len('/ZTF/raw/')
        return self.imgfile[idx:idx+4] + self.imgfile[idx+5:idx+9]

    def get_dirname(self):
        # Extrtact general data directory from image file name
        idx = self.imgfile.index('ztf_') - 21
        return self.imgfile[0:idx]

    def get_imgname(self):
        # Extrtact image file name from full link
        idx = self.imgfile.index('ztf_')
        return self.imgfile[idx:len(self.imgfile)]

    def get_ccd_cal(self):
        # Get CCD bias image computed 
        if (os.path.isfile(self.imgfile)):
            return fits.getdata(self.imgfile, 0)
        else:
            raise ValueError("-> Missing file:", self.imgfile)

    def get_ccd_bias(self):
        # Get CCD ID
        cid = self.get_ccd_id()
        cnum = int(cid)
        # Get Date
        date = self.get_date()
        # Local query
        zquery = query.ZTFQuery()
        zquery.load_metadata(kind="cal", caltype="bias", sql_query="nightdate="+str(date)+" AND ccdid="+str(cnum))
        list_bias = zquery.get_data_path(source="local")
        if (len(list_bias)!=4):
            raise ValueError("Wrong number of bias files for this CCD image:", self.imgfile)
        list_bias.sort()
        # Read CCD quadrants 
        q1 = self.read_quadrant(list_bias[0], 0)
        q2 = self.read_quadrant(list_bias[1], 0)
        q3 = self.read_quadrant(list_bias[2], 0)
        q4 = self.read_quadrant(list_bias[3], 0)
        # Each quadrant is rotated by 180° before merging
        q1 = np.rot90(q1, 2)
        q2 = np.rot90(q2, 2)
        q3 = np.rot90(q3, 2)
        q4 = np.rot90(q4, 2)
        # Horizontal merging of CCD quadrants 00 and 01 
        ccd_up = np.concatenate((q2, q1), axis=1) 
        # Horizontal merging of CCD quadrants 02 and 03
        ccd_down = np.concatenate((q3, q4), axis=1) 
        # Vertical merging of the two above half-CCD 
        ccd = np.concatenate((ccd_down, ccd_up), axis=0) 
        return ccd

    def get_flat_name(self):
        # Get Date
        date = self.get_date()
        dirdate = date[0:4]+'/'+date[4:8]+'/'
        # Get directory
        dirname = self.get_dirname()+'cal/'+dirdate+'ccdflat/'
        # Get filter ID
        fid = self.get_filter_id()
        # Get CCD ID
        cid = self.get_ccd_id()
        # Define CCD flat name
        flatname = dirname+'ztf_'+date+'_000000_'+fid+'_c'+cid+'_f.fits'
        return flatname

    def get_ccd_flat(self):
        # Check that flat exist otherwise compute and write it
        flatname = self.get_flat_name()
        if (os.path.isfile(flatname)==False):
            # Check that directory destination exist otherwise create it
            self.set_flat_directory(flatname)
            # Compute flat and write it in dirname
            self.compute_ccd_flat(flatname)
        if (os.path.isfile(flatname)):
            return fits.getdata(flatname, 0)
        else:
            raise ValueError("-> Missing CCD flat file:", imgfile)
        return ccd

    def set_flat_directory(self, flatname):
        # Get directory name
        idx = flatname.index('ztf_')
        dirname = flatname[0:idx]
        # Check if directory desdtination exists, otherwise create it
        if (os.path.isdir(dirname)):
            return
        else:
            os.system("mkdir "+dirname)
            return

    def compute_ccd_flat(self, flatname):
        date = self.get_date()
        date_bis = '/'+date[0:4]+'/'+date[4:8]+'/'
        print("|-> Compute CCD flat-field for date", date_bis)
        # Get list of raw flat fields
        list_flat = self.get_flat_list()
        # Check that list of raw flat fields is OK
        if (len(list_flat)==0):
            raise ValueError("No flat for image", self.imgfile)
        n = len(list_flat)
        fid = self.get_filter_id()
        if ((fid=='zg' or fid=='zr') and n!=20):
            raise ValueError("Wrong number of flat files for image", self.imgfile)
        if (fid=='zi' and n!=21):
            raise ValueError("Wrong number of flat files for image", self.imgfile)
        # Get bias for the whole CCD
        ccd_bias = self.get_ccd_bias()
        # Stack flat fields after bias correction
        for i in range(n):
            if (i==0):
                ccd = self.get_ccd_raw(img_file=list_flat[i]) - ccd_bias
            else:
                ccd += self.get_ccd_raw(img_file=list_flat[i]) - ccd_bias
        # Normalized flat field to median
        ccd /= np.median(ccd)
        # Write flat field
        self.image_saving(flatname, ccd)
        return True

    def get_flat_list(self):
        # Get CCD ID
        cid = self.get_ccd_id()
        cnum = int(cid)
        # Get filter ID
        fid = self.get_filter_id()
        if (fid=='zg'):
            fnum = 1
        elif (fid=='zr'):
            fnum = 2
        elif (fid=='zi'):
            fnum = 3
        else:
            fnum = 0
        # Get date
        idx = self.imgfile.index('/ZTF/raw/') + len('/ZTF/raw/')
        year = self.imgfile[idx:idx+4] 
        month = self.imgfile[idx+5:idx+7]
        day = self.imgfile[idx+7:idx+9]
        dates = [year+'-'+month+'-'+day+'T00:00:00', year+'-'+month+'-'+day+'T23:59:59']
        t = Time(dates, format='isot', scale='utc')
        dates_jd = t.jd  
        # Local query
        zquery = query.ZTFQuery()
        zquery.load_metadata(kind="raw", sql_query="imgtypecode='f' AND fid="+str(fnum)+" AND ccdid="+str(cnum)+" AND obsjd BETWEEN "+str(dates_jd[0])+" AND "+str(dates_jd[1]))
        return zquery.get_data_path(source="local")

    def compute_ccd_sci(self):
        img = self.get_imgname()
        print("|-> Compute CCD science image with global CCD flat-field corresponding to", img)
        # Get CCD raw-overscan corrected image 
        ccd_raw = self.get_ccd_raw()
        # Get CCD bias image
        ccd_bias = self.get_ccd_bias()
        # Get CCD flat image 
        ccd_flat = self.get_ccd_flat()   
        # Compute CCD science image
        ccd_corr = ccd_raw - ccd_bias
        ccd_sci = ccd_corr / ccd_flat
        # Divide CCD image in quadrant images
        quad = self.ccd_to_quadrant(ccd_sci)
        # Write each new science quadrant image with ZTF HDU
        self.write_quad(quad)
        return ccd_sci

    def get_my_sci_list(self):
        ztfsci_list = self.get_sci_list()
        # Generate my science image file names from ZTF science image file names 
        mysci_list = []
        for i in range(len(ztfsci_list)):
            img_file = ztfsci_list[i]
            idx = img_file.index('.fits')
            mysci_list.append(img_file[0:idx]+'_in2p3.fits')
        return mysci_list

    def get_ztf_ccd_sci(self):
        # Get CCD science image computed by ZTF pipeline
        #print('--> Get ZTF CCD science image for on-sky image', self.imgfile)
        imglist = self.get_sci_list()
        return self.get_ccd_sci(imglist)

    def get_my_ccd_sci(self):
        # Get CCD science image computed with the global CCD flat-field
        #print('--> Get my CCD science image for on-sky image', self.imgfile)
        mysci_list = self.get_my_sci_list()
        # Check if my CCD science image exist otherwise create it
        if (os.path.isfile(mysci_list[0]) and os.path.isfile(mysci_list[1]) and os.path.isfile(mysci_list[2]) and os.path.isfile(mysci_list[3])):
            ccd_sci = self.get_ccd_sci(mysci_list)
        else:
            # Compute my CCD science images and write it  
            ccd_sci = self.compute_ccd_sci()
        return ccd_sci

    def get_ccd_sci(self, imglist):
        # Merging of the 4 quadrants (00, 01, 02, 03) of a CCD from raw data
        # in a single image with the quadrant sub-structure
        #             q2 q1
        #             q3 q4
        # Read CCD quadrants 
        q1 = self.read_quadrant(imglist[0], 0)
        q2 = self.read_quadrant(imglist[1], 0)
        q3 = self.read_quadrant(imglist[2], 0)
        q4 = self.read_quadrant(imglist[3], 0)
        # Each quadrant is rotated by 180° before merging
        q1 = np.rot90(q1, 2)
        q2 = np.rot90(q2, 2)
        q3 = np.rot90(q3, 2)
        q4 = np.rot90(q4, 2)
        # Horizontal merging of CCD quadrants 00 and 01 
        ccd_up = np.concatenate((q2, q1), axis=1) 
        # Horizontal merging of CCD quadrants 02 and 03
        ccd_down = np.concatenate((q3, q4), axis=1) 
        # Vertical merging of the two above half-CCD 
        ccd = np.concatenate((ccd_down, ccd_up), axis=0) 
        return ccd

    def get_sci_list(self):
        # Get CCD ID
        cid = self.get_ccd_id()
        cnum = int(cid)
        # Get filter ID
        fid = self.get_filter_id()
        if (fid=='zg'):
            fnum = 1
        elif (fid=='zr'):
            fnum = 2
        elif (fid=='zi'):
            fnum = 3
        else:
            fnum = 0
        # Get field
        idx = self.imgfile.index(fid) - 7
        field = int(self.imgfile[idx:idx+6])
        # Get date
        idx = self.imgfile.index('/ZTF/raw/') + len('/ZTF/raw/')
        year = self.imgfile[idx:idx+4] 
        month = self.imgfile[idx+5:idx+7]
        day = self.imgfile[idx+7:idx+9]
        dates = [year+'-'+month+'-'+day+'T00:00:00', year+'-'+month+'-'+day+'T23:59:59']
        t = Time(dates, format='isot', scale='utc')
        dates_jd = t.jd  
        # Local query
        zquery = query.ZTFQuery()
        zquery.load_metadata(kind="sci", sql_query="fid="+str(fnum)+" AND ccdid="+str(cnum)+ "AND field="+str(field)+" AND obsjd BETWEEN "+str(dates_jd[0])+" AND "+str(dates_jd[1]))
        ztfsci_list = zquery.get_data_path(source="local", suffix="sciimg.fits")
        if (len(ztfsci_list)!=4):
            raise ValueError("Wrong number of science files for CCD image:", self.imgfile)
        return ztfsci_list

    def ccd_to_quadrant(self, ccd):
        # Divide CCD image in quadrant
        nX = int(len(ccd[0])/2)
        nY = int(len(ccd)/2)
        quad = []
        # Quadrant 01
        q1 = ccd[nY:2*nY,nX:2*nX]
        q1 = np.rot90(q1, 2)
        quad.append(q1)
        # Quadrant 02
        q2 = ccd[nY:2*nY,0:nX]
        q2 = np.rot90(q2, 2)
        quad.append(q2)
        # Quadrant 03
        q3 = ccd[0:nY,0:nX]
        q3 = np.rot90(q3, 2)
        quad.append(q3) 
        # Quadrant 04
        q4 = ccd[0:nY,nX:2*nX]
        q4 = np.rot90(q4, 2)
        quad.append(q4)
        return quad

    def write_quad(self, quad):
        # Get ZTF science image names
        ztfscilist = self.get_sci_list()
        # Get my science image names
        myscilist = self.get_my_sci_list()
        # Write each new science quadrant image with original ZTF HDU
        for i in range(len(myscilist)): 
            ztf = fits.open(ztfscilist[i])
            hdr = ztf[0].header
            os.system("rm "+myscilist[i])
            hdu = fits.PrimaryHDU(data=quad[i], header=hdr)
            hdu.writeto(myscilist[i]) 
        return 
