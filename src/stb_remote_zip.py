"""Bounded HTTP range reader for selected members of pinned large archives.

Never reads a full multi-gigabyte archive. A server ignoring Range is refused
before its body is read. No pickle or downloaded code is executed.
"""
import io,re,urllib.request,zipfile,json
from pathlib import Path
BASE='https://huggingface.co/datasets/shuinb/SP-TransientBench/resolve/cadc5d7a0fa0cf5e6f49973ba3633198c4c3c02d/'


class RangeReader(io.RawIOBase):
    def __init__(self,url,size,budget=32*1024*1024):
        self.url=url;self.size=int(size);self.position=0;self.budget=budget;self.transferred=0;self.requests=[];self.cache=[]
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        p=offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        if p<0:raise ValueError('Invalid seek')
        self.position=p;return p
    def read(self,n=-1):
        if n is None or n<0:n=self.size-self.position
        n=min(n,self.size-self.position)
        if n<=0:return b''
        if n>8*1024*1024:raise ValueError('Refusing an oversized remote read')
        start=self.position;end=start+n-1
        for a,b,data in self.cache:
            if a<=start and end<=b:
                out=data[start-a:end-a+1];self.position+=n;return out
        if self.transferred+n>self.budget:raise ValueError('Remote byte budget exhausted')
        request=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
        with urllib.request.urlopen(request,timeout=60) as response:
            if response.status!=206:raise ValueError(f'Server did not honor bounded Range:status{response.status}')
            content=response.headers.get('Content-Range','');match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',content)
            if not match or tuple(map(int,match.groups()))!=(start,end,self.size):raise ValueError('Unexpected HTTP byte range')
            if response.headers.get('Content-Length') and int(response.headers['Content-Length'])!=n:raise ValueError('Unexpected range length')
            data=response.read(n+1)
            if len(data)!=n:raise ValueError('Truncated or oversized range response')
        self.transferred+=n;self.requests.append(dict(start=start,end=end,bytes=n));self.cache.append((start,end,data))
        self.position+=n;return data


def inspect():
    root=Path(__file__).resolve().parents[1];out=root/'data'/'stb_metadata'
    archives=[('Annotations/p2/Sequence21.zip',30602679),('Histogram/p2.zip',7079798249)]
    result=[]
    for name,size in archives:
        reader=RangeReader(BASE+name,size)
        with zipfile.ZipFile(reader) as z:
            members=[dict(name=i.filename,bytes=i.file_size,compressed_bytes=i.compress_size,crc32=i.CRC,header_offset=i.header_offset) for i in z.infolist() if not i.is_dir()]
        record=dict(archive=name,archive_bytes=size,members=members,remote_bytes_read=reader.transferred,requests=reader.requests,
            provenance='Pinned HTTPS archive; only central-directory ranges read. Whole-archive hash not locally verified.')
        path=out/(name.replace('/','_')+'_members.json');path.write_text(json.dumps(record,indent=2));result.append(record)
        print(name,'members',len(members),'remote_bytes',reader.transferred,'first',[i['name'] for i in members[:5]],flush=True)
    labels=sorted(i['name'] for i in result[0]['members'] if i['name'].endswith('.npy'))
    if labels:
        selected=labels[0];stem=Path(selected).name.replace('_semantic.npy','.txt')
        matches=[i['name'] for i in result[1]['members'] if Path(i['name']).name==stem]
        print('Fixed lexical first label',selected,'histogram matches',matches,flush=True)


if __name__=='__main__':inspect()
