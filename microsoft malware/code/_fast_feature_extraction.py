######################################################
# _fast_feature_extraction.py
# author: Gert Jacobusse, gert.jacobusse@rogatio.nl
# licence: FreeBSD

"""
Copyright (c) 2015, Gert Jacobusse
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# required directory structure:
#  > feature_extraction.py
#  > trainLabels.csv
#  > sampleSubmission.csv
#  train> (all train files)
#  test>  (all test files)

######################################################
# import dependencies

import os
import csv
import zipfile
from io import BytesIO
from collections import defaultdict
import re
import numpy as np

######################################################
# list ids and labels

trainids=[]
labels=[]
with open('trainLabels.csv','r') as f:
    r=csv.reader(f)
    r.next() # skip header
    for row in r:
        trainids.append(row[0])
        labels.append(float(row[1]))

testids=[]
with open('sampleSubmission.csv','r') as f:
    r=csv.reader(f)
    r.next()
    for row in r:
        testids.append(row[0])

######################################################
# general functions

def readdata(fname,header=True,selectedcols=None):
    with open(fname,'r') as f:
        r=csv.reader(f)
        names = r.next() if header else None
        if selectedcols:
            assert header==True
            data = [[float(e) for i,e in enumerate(row) if names[i] in selectedcols] for row in r]
            names = [name for name in names if name in selectedcols]
        else:
            data = [[float(e) for e in row] for row in r]
    return data,names

def writedata(data,fname,header=None):
    with open(fname,'w') as f:
        w=csv.writer(f)
        if header:
            w.writerow(header)
        for row in data:
            w.writerow(row)

######################################################
# extract file properties

"""
function getcompressedsize
input: path to file
output: compressed size of file
* read file and compress it in memory
"""
def getcompressedsize(fpath):
    inMemoryOutputFile = BytesIO()
    zf = zipfile.ZipFile(inMemoryOutputFile, 'w') 
    zf.write(fpath, compress_type=zipfile.ZIP_DEFLATED)
    s = float(zf.infolist()[0].compress_size)
    zf.close()
    return s

"""
function writefileprops
input: ids of trainset or testset, string "train" or "test"
output: writes train_fileprops or test_fileprops
* extract file properties (size, compressed size, ratios) from all files in train or test set
"""
def writefileprops(ids,trainortest):
    with open('%s_fileprops.csv'%trainortest,'w') as f:
        w=csv.writer(f)
        w.writerow(['asmSize','bytesSize',
                    'asmCompressionRate','bytesCompressionRate',
                    'ab_ratio','abc_ratio','ab2abc_ratio'])
        for i in ids:
            asmsiz=float(os.path.getsize('%s/'%trainortest+i+'.asm'))
            bytsiz=float(os.path.getsize('%s/'%trainortest+i+'.bytes'))
            asmcr=getcompressedsize('%s/'%trainortest+i+'.asm')/asmsiz
            bytcr=getcompressedsize('%s/'%trainortest+i+'.bytes')/bytsiz
            ab=asmsiz/bytsiz
            abc=asmcr/bytcr
            w.writerow([asmsiz,bytsiz,asmcr,bytcr,ab,abc,ab/abc])
            f.flush()

######################################################
# extract asm contents

"""
the following three selections (on sections, dlls and opcodes) can be verified by looking
at the metadata files that are written during feature extraction. They are added here to
illustrate what the features mean, and to make the code more readible
"""

# sections that occur in at least 5 files from the trainset:
selsections=['.2', '.3', '.CRT', '.Lax503', '.Much', '.Pav', '.RDATA', '.Racy',
             '.Re82', '.Reel', '.Sty', '.Tls', '.adata', '.bas', '.bas0', '.brick',
             '.bss', '.code', '.cud', '.data', '.data1', '.edata', '.gnu_deb', '.hdata',
             '.icode', '.idata', '.laor', '.ndata', '.orpc', '.pdata', '.rata', '.rdat',
             '.rdata', '.reloc', '.rsrc', '.sdbid', '.sforce3', '.text', '.text1', '.tls',
             '.xdata', '.zenc', 'BSS', 'CODE', 'DATA', 'GAP', 'HEADER', 'Hc%37c',
             'JFsX_', 'UPX0', 'UPX1', 'Xd_?_mf', '_0', '_1', '_2', '_3',
             '_4', '_5', 'bss', 'code', 'seg000', 'seg001', 'seg002', 'seg003',
             'seg004']

# dlls that occur in at least 30 files from the trainset:
seldlls=['', '*', '2', '32', 'advapi32', 'advpack', 'api', 'apphelp',
         'avicap32', 'clbcatq', 'comctl32', 'comdlg32', 'crypt32', 'dbghelp', 'dpnet', 'dsound',
         'e', 'gdi32', 'gdiplus', 'imm32', 'iphlpapi', 'kernel32', 'libgcj_s', 'libvlccore',
         'mapi32', 'mfc42', 'mlang', 'mpr', 'msasn1', 'mscms', 'mscoree', 'msdart',
         'msi', 'msimg32', 'msvcp60', 'msvcp71', 'msvcp80', 'msvcr71', 'msvcr80', 'msvcr90',
         'msvcrt', 'msvfw32', 'netapi32', 'ntdll', 'ntdsapi', 'ntmarta', 'ntshrui', 'ole32',
         'oleacc', 'oleaut32', 'oledlg', 'opengl32', 'psapi', 'rasapi32', 'riched20', 'riched32',
         'rnel32', 'rpcrt4', 'rsaenh', 'secur32', 'security', 'sensapi', 'setupapi', 'shell32',
         'shfolder', 'shlwapi', 'tapi32', 'unicows', 'urlmon', 'user32', 'usp10', 'uxtheme',
         'version', 'wab32', 'wininet', 'winmm', 'wintrust', 'wldap32', 'ws2_32', 'wsock32',
         'xprt5']

# opcodes that occur in at least 30 files from the trainset:
selopcs=['aad', 'aam', 'adc', 'add', 'addpd', 'addps', 'addsd', 'align',
        'and', 'andnps', 'andpd', 'andps', 'arpl', 'assume', 'bound', 'bsf',
        'bsr', 'bswap', 'bt', 'btc', 'btr', 'bts', 'call', 'cmova',
        'cmovb', 'cmovbe', 'cmovg', 'cmovge', 'cmovl', 'cmovle', 'cmovnb', 'cmovns',
        'cmovnz', 'cmovs', 'cmovz', 'cmp', 'cmpeqsd', 'cmpltpd', 'cmps', 'cmpxchg',
        'db', 'dd', 'dec', 'div', 'divsd', 'dq', 'dt', 'dw',
        'end', 'endp', 'enter', 'fadd', 'faddp', 'fbld', 'fbstp', 'fcmovb',
        'fcmovbe', 'fcmove', 'fcmovnb', 'fcmovnbe', 'fcmovne', 'fcmovnu', 'fcmovu', 'fcom',
        'fcomi', 'fcomip', 'fcomp', 'fdiv', 'fdivp', 'fdivr', 'fdivrp', 'ffree',
        'ffreep', 'fiadd', 'ficom', 'ficomp', 'fidiv', 'fidivr', 'fild', 'fimul',
        'fist', 'fistp', 'fisttp', 'fisub', 'fisubr', 'fld', 'fldcw', 'fldenv',
        'fmul', 'fmulp', 'fnsave', 'fnstcw', 'fnstenv', 'fnstsw', 'frstor', 'fsave',
        'fst', 'fstcw', 'fstp', 'fstsw', 'fsub', 'fsubp', 'fsubr', 'fsubrp',
        'fucom', 'fucomi', 'fucomip', 'fucomp', 'fxch', 'hnt', 'hostshort',
        'ht', 'idiv', 'imul', 'in', 'inc', 'include', 'int', 'ja', 'jb', 'jbe',
        'jecxz', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jnb', 'jno', 'jnp', 'jns',
        'jnz', 'jo', 'jp', 'js', 'jz', 'ldmxcsr', 'lds', 'lea', 'les', 'lock',
        'lods', 'loop', 'loope', 'loopne', 'mov', 'movapd', 'movaps', 'movd',
        'movdqa', 'movhps', 'movlpd', 'movlps', 'movq', 'movs', 'movsd', 'movss',
        'movsx', 'movups', 'movzx', 'mul', 'mulpd', 'mulps', 'mulsd', 'neg',
        'nop', 'not', 'offset', 'or', 'orpd', 'orps', 'out', 'outs', 'paddb',
        'paddd', 'paddq', 'paddsb', 'paddsw', 'paddusb', 'paddusw', 'paddw',
        'pand', 'pandn', 'pavgb', 'pcmpeqb', 'pcmpeqd', 'pcmpeqw', 'pcmpgtb',
        'pcmpgtd', 'pcmpgtw', 'pextrw', 'piconinfo', 'pinsrw', 'pmaddwd',
        'pmaxsw', 'pmulhw', 'pmullw', 'pop', 'por', 'pperrinfo', 'proc',
        'pshufd', 'pshufw', 'pslld', 'psllq', 'psllw', 'psrad', 'psraw',
        'psrld', 'psrlq', 'psrlw', 'psubb', 'psubd', 'psubq', 'psubsb',
        'psubsw', 'psubusb', 'psubusw', 'psubw', 'public', 'punpckhbw',
        'punpckhdq', 'punpckhwd', 'punpcklbw', 'punpckldq', 'punpcklwd',
        'push', 'pxor', 'rcl', 'rcpps', 'rcr', 'rep', 'repe', 'repne',
        'retf', 'retfw', 'retn', 'retnw', 'rgsabound', 'rol', 'ror', 'sal',
        'sar', 'sbb', 'scas', 'segment', 'setb', 'setbe', 'setl', 'setle',
        'setnb', 'setnbe', 'setnl', 'setnle', 'setns', 'setnz', 'seto',
        'sets', 'setz', 'shl', 'shld', 'shr', 'shrd', 'shufps', 'sldt',
        'stmxcsr', 'stos', 'sub', 'subpd', 'subps', 'subsd', 'test',
        'ucomisd', 'unicode', 'xadd', 'xchg', 'xlat', 'xor', 'xorpd', 'xorps']

"""
function getsectioncounts
input: list of lines in an asm file
output: dictionary with number of lines in each section
* count number of lines in each section
"""
def getsectioncounts(asmlines):
    sectioncounts=defaultdict(int)
    for l in asmlines:
        sectioncounts[l.split(':')[0]]+=1
    return sectioncounts

"""
function getcalls
input: list of lines in an asm file
output: dictionary with number of times each function is found
* count number of times each function occurs
"""
def getcalls(asmlines):
    calls=defaultdict(int)
    for l in asmlines:
        ls=l.split('__stdcall ')
        if len(ls)>1:
            calls[ls[1].split('(')[0]]+=1
    return calls

"""
function getdlls
input: list of lines in an asm file
output: dictionary with number of times each dll is found
* count number of times each dll occurs
"""
def getdlls(asmlines):
    dlls=defaultdict(int)
    for l in asmlines:
        ls=l.lower().split('.dll')
        if len(ls)>1:
            dlls[ls[0].replace('\'',' ').split(' ')[-1].split('"')[-1].split('<')[-1].split('\\')[-1].split('\t')[-1]]+=1
    return dlls

"""
function getopcodeseries
input: list of lines in an asm file
output: series of opcodes in the order in which they occur
* extract all opcodes using regular expressions
* first used to create opcode ngrams, but later translated to counts using series2freqs
"""
def getopcodeseries(asmlines):
    ops=[]
    opex=re.compile('(   )([a-z]+)( )')
    for l in asmlines:
        e=opex.search(l)
        if e:
            ops.append(e.group(2))
    return ops

def series2freqs(series):
    freqs=defaultdict(int)
    for e in series:
        freqs[e]+=1
    return freqs

"""
function getqperc
input: list of lines in an asm file
output: percent of characters that is a questionmark
* count number of questionmarks and divide it by number of characters
"""
def getqperc(asmlines):
    n=0
    nq=0
    for l in asmlines:
        for c in l:
            n+=1
            if c=='?':
                nq+=1
    return float(nq)/n

"""
function countbysection
input: list of lines in an asm file, list of sections to include, list of characters to include
output: number of occurences of each specified character by section, list of feature names, list of characters included
* count number of occurences of each specified character by section
"""
def countbysection(asmlines,segms,chars=[' ','?','.',',',':',';','+','-','=','[','(','_','*','!','\\','/','\''],namesonly=False):
    names=['%s_tot'%ss for ss in selsections]
    for c in chars:
        names.extend(['%s_c%s'%(ss,c) for ss in selsections])
    if namesonly:
        return names+['restsegm']+['%s_restchar'%ss for ss in selsections]
    ns=len(segms)
    nc=len(chars)
    segmdict={e:i for i,e in enumerate(segms)}
    chardict={e:i for i,e in enumerate(chars)}
    counts=[0 for i in xrange((nc+1)*ns)]
    for l in asmlines:
        segm=l.split(':')[0]
        if segm in segmdict:
            s=segmdict[segm]
            for ch in l:
                counts[s]+=1
                if ch in chardict:
                    c=chardict[ch]
                    counts[ns+c*ns+s]+=1
    return counts,names,chars

"""
function normalizecountbysection
input: output of function countbysection
output: normalized number of occurences of each specified character by section
* divide number of occurences of each specified character by section by total number
* and calculate rest percent of other characters and other sections
"""
def normalizecountbysection(counts,names,chars):
    d={names[i]:counts[i] for i in xrange(len(names))}
    tot=sum([d['%s_tot'%s] for s in selsections])
    d['restsegm']=1.0
    for s in selsections:
        d['%s_restchar'%s]=0.0
        if d['%s_tot'%s] > 0:
            d['%s_restchar'%s]=1.0
            for c in chars:
                d['%s_c%s'%(s,c)]=float(d['%s_c%s'%(s,c)])/d['%s_tot'%s]
                d['%s_restchar'%s]-=d['%s_c%s'%(s,c)]
            d['%s_tot'%s]=float(d['%s_tot'%s])/tot
            d['restsegm']-=d['%s_tot'%s]
    return [d[name] for name in names+['restsegm']+['%s_restchar'%ss for ss in selsections]]

"""
function writeasmcontents
input: ids of trainset or testset, string "train" or "test"
output: writes train_asmcontents or test_asmcontents + metadata on sections, calls, dlls and opcodes
* extract features from contents of asm from all files in train or test set
* by reading list of asm lines from each file and calling the previous functions
"""
def writeasmcontents(ids,trainortest):
    with open('%s_asmcontents.csv'%trainortest,'w') as f:
        w=csv.writer(f)
        w.writerow(
            ['sp_%s'%key for key in selsections]
            +['dl_%s'%key for key in seldlls]
            +['op_%s'%key for key in selopcs]
            +['qperc']
            +countbysection(None,selsections,namesonly=True)
            )
        fsec=open('secmetadata%s.txt'%trainortest,'w')
        wsec=csv.writer(fsec)
        fcal=open('calmetadata%s.txt'%trainortest,'w')
        wcal=csv.writer(fcal)
        fdll=open('dllmetadata%s.txt'%trainortest,'w')
        wdll=csv.writer(fdll)
        fopc=open('opcmetadata%s.txt'%trainortest,'w')
        wopc=csv.writer(fopc)
        for i in ids:
            with open('%s/'%trainortest+i+'.asm','r') as fasm:
                asmlines=[line for line in fasm.readlines()]
            # section counts/ proportions
            sc=getsectioncounts(asmlines)
            wsec.writerow([i]+['%s:%s'%(key,sc[key]) for key in sc if sc[key]>0])
            scsum=sum([sc[key] for key in sc])
            secfeat=[float(sc[key])/scsum for key in selsections]
            # calls
            cal=getcalls(asmlines)
            wcal.writerow([i]+['%s:%s'%(key,cal[key]) for key in cal if cal[key]>0])
            # dlls
            dll=getdlls(asmlines)
            wdll.writerow([i]+['%s:%s'%(key,dll[key]) for key in dll if dll[key]>0])
            dllfeat=[float(dll[key]) for key in seldlls]
            # opcodes
            opc=series2freqs(getopcodeseries(asmlines))
            wopc.writerow([i]+['%s:%s'%(key,opc[key]) for key in opc if opc[key]>0])
            opcfeat=[float(opc[key]) for key in selopcs]
            # overall questionmark proportion
            qperc=getqperc(asmlines)
            # normalized interpunction characters by section
            ipbysecfeat=normalizecountbysection(*countbysection(asmlines,selsections))
            #
            w.writerow(secfeat+dllfeat+opcfeat+[qperc]+ipbysecfeat)
            f.flush()
        fsec.close()
        fcal.close()
        fdll.close()
        fopc.close()

######################################################
# reduce asm contents features, using a criterion on the number of files with nonzero value

"""
function writeasmcontents
input: traindata matrix, testdata matrix, feature names, criterion on required number of nonzeros in each column
output: reduced traindata matrix, testdata matrix and feature names
* calculate number of nonzeros by column and keep only features that meet the criterion
"""
def reducefeatures(xtrain,xtest,names,ncrit=500):
    ntrain=np.sum(np.array([np.array([ei!=0 for ei in e]) for e in xtrain]),axis=0)
    xtrain=np.array([np.array([e[j] for j,n in enumerate(ntrain) if n>ncrit]) for e in xtrain])
    xtest=np.array([np.array([e[j] for j,n in enumerate(ntrain) if n>ncrit]) for e in xtest])
    names=[names[j] for j,n in enumerate(ntrain) if n>ncrit]
    return xtrain,xtest,names

"""
function reduceasmcontents
input: none
output: write reduced asm contents
* read features on asm contents, reduce them by calling reducefeatures and write the results
"""
def reduceasmcontents():
    train_asmcontents,asmcontentshead=readdata('train_asmcontents.csv')
    test_asmcontents,_=readdata('test_asmcontents.csv')
    train_asmcontents_red,test_asmcontents_red,asmcontentshead_red=reducefeatures(
                                        train_asmcontents,test_asmcontents,asmcontentshead)
    writedata(train_asmcontents_red,'train_asmcontents_red.csv',asmcontentshead_red)
    writedata(test_asmcontents_red,'test_asmcontents_red.csv',asmcontentshead_red)

######################################################
# calculate statistics on asm metadata

"""
function loadmetadata
input: path of metadatafile (written by writeasmcontents)
output: dictionary with metadata
* load metadata into dictionary
"""
def loadmetadata(inpath):
    md={}
    with open(inpath,'r') as f:
        r=csv.reader(f)
        for row in r:
            md[row[0]]=defaultdict(int)
            for e in row[1:]:
                key,value=e.split(':')[-2:]
                md[row[0]][key]=int(value)
    return md

"""
function getstats
input: metadata dictionary, dictionary keys sorted by number of occurrences over train and test set, type of metadata (sec[tions], dll[s], cal[ls] or opc[odes])
output: statistics on the specified type of metadata
* calculate statistics on the specified type of metadata
"""
def getstats(dct,sortedkeys,datatype):
    stats={}
    for i in dct:
        stats[i]={}
        d=dct[i]
        n=len(d)
        sm=sum([d[key] for key in d]) if n>0 else 0
        pmx=100*max([d[key] for key in d])/sm if n>0 else 0
        stats[i]['%s_nkey'%datatype]=n
        stats[i]['%s_sum'%datatype]=sm
        stats[i]['%s_pmax'%datatype]=pmx
        top5={key for key in sortedkeys[:5]}
        top20={key for key in sortedkeys[:20]}
        top50={key for key in sortedkeys[:50]}
        stats[i]['%s_n5key'%datatype]=len([e for e in d if e in top5])
        stats[i]['%s_n20key'%datatype]=len([e for e in d if e in top20])
        stats[i]['%s_n50key'%datatype]=len([e for e in d if e in top50])
    return stats

"""
function getstatsfromdata
input: type of metadata (sec[tions], dll[s], cal[ls] or opc[odes])
output: statistics on the specified type of metadata for both trainset and testset
* for each value, count the number of occurrences over train and test set
* call getstats to calculate statistics for each file
"""
def getstatsfromdata(datatype):
    traindict=loadmetadata('%smetadatatrain.txt'%datatype)
    testdict=loadmetadata('%smetadatatest.txt'%datatype)
    allkeys=defaultdict(int)
    for i in traindict:
        for key in traindict[i]:
            allkeys[key]+=1
    for i in testdict:
        for key in testdict[i]:
            allkeys[key]+=1       
    sortedkeys=sorted([key for key in allkeys],reverse=True,key=lambda x: allkeys[x])
    trainstats=getstats(traindict,sortedkeys,datatype)
    teststats=getstats(testdict,sortedkeys,datatype)
    return trainstats,teststats

"""
function writeasmstats
input: feature types (use only default; why is this a parameter??? I don't remember)
output: writes asm statistics to files for both train and test set
* call getstatsfromdata to calculate statistics on sec[tions], dll[s], cal[ls] and opc[odes]
* write the results to train_asmstats and test_asmstats
"""
def writeasmstats(stats=['nkey', 'sum', 'pmax', 'n5key', 'n20key', 'n50key']):
    traindata=[[] for i in xrange(len(trainids))]
    testdata=[[] for i in xrange(len(testids))]
    names=[]
    for datatype in ['sec','dll','cal','opc']:
        keys=['%s_%s'%(datatype,stat) for stat in stats]
        trainstats,teststats=getstatsfromdata(datatype)
        for inum,i in enumerate(trainids):
            traindata[inum].extend([trainstats[i][key] for key in keys])
        for inum,i in enumerate(testids):
            testdata[inum].extend([teststats[i][key] for key in keys])
        names.extend(keys)
    writedata(traindata,'train_asmstats.csv',names)
    writedata(testdata,'test_asmstats.csv',names)

######################################################
# extract bytes contents

"""
function getcompressedsize_str
input: string
output: compressed size of string
* compress string and return the compressed size
"""
def getcompressedsize_str(strinput):
    inMemoryOutputFile = BytesIO()
    zf = zipfile.ZipFile(inMemoryOutputFile, 'w')
    zf.writestr('',strinput, compress_type=zipfile.ZIP_DEFLATED)
    return zf.infolist()[0].compress_size

"""
function writeblocksizes
input: ids of trainset or testset, string "train" or "test"
output: writes train_blocksizes or test_blocksizes
* calculate and write compressed size of each 4 kB block for all files in train or test set
"""
def writeblocksizes(ids,trainortest,blocksize=256): # 256 times 16 bytes = 4096 bytes
    with open('%s_blocksizes.csv'%trainortest,'w') as fout:
        for i in ids:
            with open('%s/%s.bytes'%(trainortest,i),'r') as fin:
                contents=fin.readlines()
            fout.write('%s,'%i)
            n=len(contents)
            blocksize=256 
            nblock=n/256
            for b in xrange(nblock):
                strinput=''
                for lidx in xrange(b*blocksize,(b+1)*blocksize):
                    l=contents[lidx]
                    strinput += l[l.find(' ')+1:-1]
                s=getcompressedsize_str(strinput)
                fout.write('%d,'%s)
            fout.write('\n')

"""
function writeblocksizedistributions
input: string "train" or "test"
output: writes train_blocksizedistributions or test_blocksizedistributions
* calculate statistics on files with blocksizes to get the same number of features for each file
"""
def writeblocksizedistributions(trainortest):
    with open('%s_blocksizes.csv'%trainortest,'r') as f:
        with open('%s_blocksizedistributions.csv'%trainortest,'w') as fout:
            fout.write('cs4k_min,cs4k_p10,cs4k_p20,cs4k_p30,cs4k_p50,cs4k_p70,cs4k_p80,cs4k_p90,cs4k_max,cs4k_mean,cs4k_q1mean,cs4k_q2mean,cs4k_q3mean,cs4k_q4mean\n')
            for i,l in enumerate(f):
                ls=l.split(',')
                sizes=[float(e) for e in ls[1:-1]]
                slen=len(sizes)
                qlen=1 if slen/4<1 else slen/4
                q1m=np.mean(sizes[:qlen])
                q2m=np.mean(sizes[qlen:2*qlen])
                q3m=np.mean(sizes[-2*qlen:-qlen])
                q4m=np.mean(sizes[-qlen:])
                sizes=sorted(sizes)
                maxidx=slen-1
                fout.write('%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f\n'%(
                    sizes[0],
                    sizes[10*maxidx//100],
                    sizes[20*maxidx//100],
                    sizes[30*maxidx//100],
                    sizes[50*maxidx//100],
                    sizes[70*maxidx//100],
                    sizes[80*maxidx//100],
                    sizes[90*maxidx//100],
                    sizes[-1],
                    round(np.mean(sizes)),
                    q1m,q2m,q3m,q4m))

######################################################
# build combined train and test files

"""
function writecombifile
input: list of lists with 2 entries: file name and optional feature names (None for all), filename to write results to
output: writes file with combined feature sets
* combine features from different sets into a single file
* ids, labels and header are optional
"""
def writecombifile(sourcefilesandselections,filename,includeid=True,includelabel=True,header=True):
    nsource=len(sourcefilesandselections)
    for trainortest in ['train','test']:
        alldata=[]
        allnames=[]
        for source,selection in sourcefilesandselections:
            data,names=readdata('%s_%s'%(trainortest,source),selectedcols=selection)
            alldata.append(data)
            allnames.extend(names)
        with open('%s_%s'%(trainortest,filename),'w') as f:
            w=csv.writer(f)
            if header:
                w.writerow((['Id'] if includeid else [])+
                            allnames+
                            (['Class'] if includelabel else []))
            ids = trainids if trainortest=='train' else testids
            for inum,i in enumerate(ids):
                datarow=[]
                for src in xrange(nsource):
                    datarow.extend(alldata[src][inum])
                w.writerow(
                    ([i] if includeid else [])+
                    datarow+
                    ([labels[inum] if trainortest=='train' else -1] if includelabel else []))
            
######################################################
# go

if __name__ == '__main__':
    writefileprops(trainids,'train')
    writefileprops(testids,'test')
    writeasmcontents(trainids,'train')
    writeasmcontents(testids,'test')
    reduceasmcontents()
    writeasmstats()
    writeblocksizes(trainids,'train')
    writeblocksizes(testids,'test')
    writeblocksizedistributions('train')
    writeblocksizedistributions('test')
    writecombifile(
        (
        ['fileprops.csv',None],
        ['asmcontents.csv',
                ['sp_%s'%key for key in selsections]
                +['dl_%s'%key for key in seldlls]
                +['op_%s'%key for key in selopcs]
                +['qperc']
                +[e for e in countbysection(None,selsections,namesonly=True) if e.endswith('_c?')]
                 ],
        ),
        '20.csv',
        includeid=False,
        includelabel=False,
        header=False
        )
    writecombifile(
        (
        ['fileprops.csv',None],
        ['asmcontents.csv',
                ['dl_%s'%key for key in seldlls]
                +['op_%s'%key for key in selopcs]
                +['qperc']
                +countbysection(None,selsections,namesonly=True)
                 ],
        ),
        '28_std.csv',
        includeid=False,
        includelabel=False,
        header=False
        )
    writecombifile(
        (
        ['fileprops.csv',None],
        ['asmcontents_red.csv',
                ['dl_%s'%key for key in seldlls]
                +['op_%s'%key for key in selopcs]
                +['qperc']
                +countbysection(None,selsections,namesonly=True)
                 ],
        ['asmstats.csv',None],
        ['blocksizedistributions.csv',None],
        ),
        '45c.csv',
        includeid=False,
        includelabel=False
        )

