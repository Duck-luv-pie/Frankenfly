from companion_brain.body.eyes import eyes_for, POSES
from companion_brain.body.decode import Decoded

def test_angry_mirrors_lids():
    p=eyes_for(Decoded(state='threat'),0,0)
    assert p['l']['tilt']>0 and p['r']['tilt']==-p['l']['tilt']
    assert p['l']['ut']<.5

def test_search_squints_scans_and_holds():
    poses=[eyes_for(Decoded(),0,0,expression='searching',now=t)['l'] for t in [.3,.8,1.5,2.7]]
    assert poses[0]['px']==poses[1]['px']
    assert len({p['px'] for p in poses})==3
    assert all(p['ut']<.5 and p['lt']<.5 for p in poses)

def test_all_blinks_close_both_lids():
    for state in POSES:
        p=eyes_for(Decoded(state=state),0,0,blink=True,now=0)
        assert all(e['ut']==e['lt']==e['tilt']==0 for e in p.values())
