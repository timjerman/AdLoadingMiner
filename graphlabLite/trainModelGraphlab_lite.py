# -*- coding: utf-8 -*-
"""
Created on Tue May 31 18:07:49 2016

@author: TimJ
"""

import graphlab
import graphlab.aggregate as agg
import numpy as np

def trainModel(trainDataFile,valDataFile, modelSavePath, outlierProbability):

    #Prepare data
    def prepareData(data, perct, isTestDataset = False):

        if not isTestDataset:   
            data['ADLOADINGTIME']=data['ADLOADINGTIME'].astype(float)
            print(np.percentile(data['ADLOADINGTIME'],perct))
            if perct < 100:
                data = data[data['ADLOADINGTIME']<np.percentile(data['ADLOADINGTIME'],perct)] 

        #convert to int types
        data['GEOIP_LAT'] = data['GEOIP_LAT'].apply(lambda x: '0' if x=='null' else x)
        data['GEOIP_LAT']=data['GEOIP_LAT'].astype(float)
        data['GEOIP_LNG'] = data['GEOIP_LNG'].apply(lambda x: '0' if x=='null' else x)
        data['GEOIP_LNG']=data['GEOIP_LNG'].astype(float)
        data['UA_MOBILEDEVICE'] = data['UA_MOBILEDEVICE'].apply(lambda x: '0' if x=='null' else x)
        data['UA_MOBILEDEVICE']=data['UA_MOBILEDEVICE'].astype(int)    
        data['TIMESTAMP']=data['TIMESTAMP'].astype(int)
        data['HOSTWINDOWHEIGHT'] = data['HOSTWINDOWHEIGHT'].apply(lambda x: '0' if x=='null' else x)
        data['HOSTWINDOWHEIGHT']=data['HOSTWINDOWHEIGHT'].astype(int)
        data['HOSTWINDOWWIDTH'] = data['HOSTWINDOWWIDTH'].apply(lambda x: '0' if x=='null' else x)
        data['HOSTWINDOWWIDTH']=data['HOSTWINDOWWIDTH'].astype(int)
        data['TOPMOSTREACHABLEWINDOWHEIGHT'] = data['TOPMOSTREACHABLEWINDOWHEIGHT'].apply(lambda x: '0' if x=='null' else x)
        data['TOPMOSTREACHABLEWINDOWHEIGHT']=data['TOPMOSTREACHABLEWINDOWHEIGHT'].astype(int)
        data['TOPMOSTREACHABLEWINDOWWIDTH'] = data['TOPMOSTREACHABLEWINDOWWIDTH'].apply(lambda x: '0' if x=='null' else x)
        data['TOPMOSTREACHABLEWINDOWWIDTH']=data['TOPMOSTREACHABLEWINDOWWIDTH'].astype(int)


        data['TOPMOSTREACHABLEWINDOWAREA'] = data['TOPMOSTREACHABLEWINDOWWIDTH']*data['TOPMOSTREACHABLEWINDOWHEIGHT'];

        data['GEOIP_COUNTRY'] = data['GEOIP_COUNTRY'] + '_' + data['GEOIP_REGION'] 
        data['GEOIP_COUNTRY'] = data['GEOIP_COUNTRY'].apply(lambda x: x if 'United States' in x.split('_')[0] else x.split('_')[0])

        data['ERRORSJSON'] = data['ERRORSJSON'].apply(lambda x: x.replace('"',''))
        data['FILESJSON'] = data['FILESJSON'].apply(lambda x: x.replace('"',''))

        def extractImageSize(sIn):
            sIn = sIn.replace('[','')
            sIn = sIn.replace('}','')
            sIn = sIn.replace(']','')
            sIn = sIn.replace('{type:image,size:','')
            numberList = [int(s) for s in sIn.split(',') if s.isdigit()]
            if len(numberList) == 0:
                numberList = 0
            return np.sum(numberList)

        def extractVideoSize(sIn):
            sIn = sIn.replace('[','')
            sIn = sIn.replace('}','')
            sIn = sIn.replace(']','')
            sIn = sIn.replace('{type:video,size:','')
            numberList = [int(s) for s in sIn.split(',') if s.isdigit()]
            if len(numberList) == 0:
                numberList = 0
            return np.sum(numberList)

        data['FILESJSON_IMGSIZE'] = data['FILESJSON'].apply(extractImageSize)
        data['FILESJSON_VIDSIZE'] = data['FILESJSON'].apply(extractVideoSize)

        data['COMBINEDID'] = data['ACCOUNTID']+data['CAMPAIGNID']+data['CREATIVEID']
        data['COMBINEDEXTERNALID'] = data['EXTERNALPLACEMENTID']+data['EXTERNALSITEID']+data['EXTERNALSUPPLIERID']    

        data['PLATFORMCOMBINED'] = data['PLATFORM']+data['PLATFORMVERSION']
        data['PLATFORMCOMBINED'] = data['PLATFORMCOMBINED'].apply(lambda x : x.replace('null', ''))
        data['PLATFORMCOMBINED'] = data['PLATFORMCOMBINED'].apply(lambda x : 'NA' if x == '' else x)

        data['UA_OSCOMB'] = data['UA_OS']+data['UA_OSVERSION']


        data['INTENDENTISACTUALDEVICETYPE'] = data['INTENDEDDEVICETYPE']==data['ACTUALDEVICETYPE']
        data['PLATFORMCOMBINEDISOSCAOMB'] = data['PLATFORMCOMBINED']==data['UA_OSCOMB']


        # remove unneeded columns

        #data.remove_columns(['UA_OS', 'UA_OSVERSION']);
        #data.remove_columns(['PLATFORM','PLATFORMVERSION']);
        #data.remove_columns(['UA_PLATFORM','UA_PLATFORMVERSION']);
        #data.remove_columns(['CDNNAME','UA_DEVICETYPE','ACTUALDEVICETYPE']);
        #data.remove_column('EXTERNALPLACEMENTID');
        #data.remove_column('EXTERNALSITEID');
        #data.remove_column('EXTERNALSUPPLIERID');
        #data.remove_column('ACCOUNTID');
        #data.remove_column('CAMPAIGNID');
        #data.remove_column('CREATIVEID');
        #data.remove_column('TOPMOSTREACHABLEWINDOWWIDTH');
        #data.remove_column('TOPMOSTREACHABLEWINDOWHEIGHT');
        #data.remove_column('HOSTWINDOWWIDTH');
        #data.remove_column('HOSTWINDOWHEIGHT');
        #data.remove_column('DEVICEORIENTATION');
        #data.remove_column('UA_MOBILEDEVICE');
        #data.remove_column('EXTERNALCREATIVEID');
        #data.remove_column('NETWORKTYPE');
        #data.remove_column('GEOIP_TIMEZONE');
        #data.remove_column('GEOIP_METROCODE');
        #data.remove_column('GEOIP_AREACODE');
        #data.remove_column('GEOIP_CITY');
        #data.remove_column('GEOIP_DMACODE');
        #data.remove_column('GEOIP_REGION');

        return data

    sep = '\t'

    train_data = graphlab.SFrame.read_csv(trainDataFile,sep,column_type_hints=str);
    train_data = prepareData(train_data, outlierProbability)

    val_data = graphlab.SFrame.read_csv(valDataFile,sep,column_type_hints=str);
    val_data = prepareData(val_data, outlierProbability)

    features = [
     'PLACEMENTID',
     'TIMESTAMP',
     'CREATIVETYPE',
     'UA_HARDWARETYPE',
     'UA_VENDOR',
     'UA_MODEL',
     'UA_BROWSER',
     'UA_BROWSERVERSION',
     'FILESJSON',
     'ERRORSJSON',
     'TOPMOSTREACHABLEWINDOWAREA',
     'FILESJSON_IMGSIZE',
     'FILESJSON_VIDSIZE',
     'COMBINEDID',
     'COMBINEDEXTERNALID',
     'PLATFORMCOMBINED',
     'UA_OSCOMB',
     'SDK',
     'EXTERNALADSERVER'
               ]

    model = graphlab.random_forest_regression.create(train_data, target='ADLOADINGTIME',validation_set=val_data,
                                               max_iterations=25, features=features,
                                               max_depth =  50, row_subsample=0.9, column_subsample = 0.6)

    model.save(modelSavePath)
    
if __name__ == '__main__':
 
    trainDataFile = 'E:/celtraMiningFrik/ccdm_large.tsv'
    valDataFile = 'E:/ccdm_medium/ccdm_large.tsv'
    modelSavePath = 'E:/celtraMiningFrik/RFReg'
    outlierProbability = 95.25
    
    trainModel(trainDataFile,valDataFile, modelSavePath, outlierProbability)