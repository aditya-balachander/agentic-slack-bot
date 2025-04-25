import os
from string import Template
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

""" SfdcOrg object holds login information and provides below information
      - Session ID
      - Org URL
"""
class SfdcOrg:
    def __init__(self,loginURL,usernane,password):
        self.loginURL=loginURL
        self.username=usernane
        self.password=password
        self.__serverURL=""
        self.__serverHostname=""
        self.__sessionId=""
        self.__sessionValidTill=datetime.now()
        self.proxies = {}
        if "USE_PROXIES" in os.environ:
            if os.environ["USE_PROXIES"].lower() == "true":
                self.proxies = {
                    'http': 'http://public0-proxy1-0-xrd.data.sfdc.net:8080',
                    'https': 'http://public0-proxy1-0-xrd.data.sfdc.net:8080'
                }

    # Login into Org and get session ID
    def __loginOrg(self):
        loginHeaderTemplate=Template("""
            <env:Envelope xmlns:xsd="http://www.w3.org/2001/XMLSchema"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xmlns:env="http://schemas.xmlsoap.org/soap/envelope/">
            <env:Body>
                <n1:login xmlns:n1="urn:partner.soap.sforce.com">
                <n1:username>$orgUsername</n1:username>
                <n1:password>$orgPassword</n1:password>
                </n1:login>
            </env:Body>
            </env:Envelope>
            """)
        loginEnvelope=loginHeaderTemplate.substitute(orgUsername=self.username,orgPassword=self.password)
        requestHeaders = {'content-type': 'text/xml; charset: UTF-8', 'SOAPAction': 'login'}
        response = requests.post(self.loginURL+"/services/Soap/u/47.0",data=loginEnvelope,headers=requestHeaders,proxies=self.proxies)
        #print(response.status_code)
        if(response.status_code == 200):
            self.__serverURL = ET.fromstring(response.text).findall('./{http://schemas.xmlsoap.org/soap/envelope/}Body/{urn:partner.soap.sforce.com}loginResponse/{urn:partner.soap.sforce.com}result/{urn:partner.soap.sforce.com}metadataServerUrl')[0].text
            self.__serverHostname = ET.fromstring(response.text).findall('./{http://schemas.xmlsoap.org/soap/envelope/}Body/{urn:partner.soap.sforce.com}loginResponse/{urn:partner.soap.sforce.com}result/{urn:partner.soap.sforce.com}metadataServerUrl')[0].text.split('/')[2]
            self.__sessionId = ET.fromstring(response.text).findall('./{http://schemas.xmlsoap.org/soap/envelope/}Body/{urn:partner.soap.sforce.com}loginResponse/{urn:partner.soap.sforce.com}result/{urn:partner.soap.sforce.com}sessionId')[0].text
            validitySeconds=ET.fromstring(response.text).findall('./{http://schemas.xmlsoap.org/soap/envelope/}Body/{urn:partner.soap.sforce.com}loginResponse/{urn:partner.soap.sforce.com}result/{urn:partner.soap.sforce.com}userInfo/{urn:partner.soap.sforce.com}sessionSecondsValid')[0].text
            self.__sessionValidTill=datetime.now()+timedelta(seconds=int(validitySeconds))
        else:
            print("Unable to login into org")
            raise SignInError("Unable to login into org, response code: " + str(response.status_code))

    # Returns session ID
    def getSessionId(self):
        if(self.__sessionId == "" or self.__sessionValidTill <= datetime.now()):
            self.__loginOrg()
        return(self.__sessionId)

    # Returns org hostname
    def getServerHostname(self):
        if(self.__serverHostname == "" or self.__sessionValidTill <= datetime.now()):
            self.__loginOrg(self)
        return(self.__serverHostname)

class SignInError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)