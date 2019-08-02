import json
import logging
import traceback
import json
import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
import requests
import datetime
import sys
from gensim.parsing.preprocessing import preprocess_string

def alert_mail_send(to, subject, text):
    try:
        resp = requests.post(url='http://10.0.6.146:3009/send',
                             data={
                                 "to": to,
                                 "subject": subject,
                                 "text": text,
                             })

        resp_json = json.loads(resp.text)
        logging.getLogger(__file__).info(resp_json)
        if resp_json['status'] == 'success':
            return True, None
        else:
            return False, str(resp_json)
    except Exception as err:
        traceback.print_exc()
        return False, str(err)

if __name__ == "__main__":
    """
    alert_mail_send("sicheng.tang@tigerobo.com", "中文", "233")
    a = json.dumps("hi")
    print(a)
    print(type(a))
    """
    print(preprocess_string("<i>Hel 9lo</i> <b>Wo9 rld</b>! Th3     weather_is really g00d today, isn't it?"))




def send_mail(send_from, send_to, subject, text, files=None,
              server="127.0.0.1"):
    assert isinstance(send_to, list)

    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(text))

    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
        msg.attach(part)


    smtp = smtplib.SMTP(server)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()
