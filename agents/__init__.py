from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent
from agents.dfp import DFPAgent
from agents.dfm import DFMAgent
from agents.dsrl import DSRLAgent
from agents.grpo import GRPOAgent
from agents.svgd import SVGDAgent
from agents.mfp import MFPAgent
from agents.qam import QAMAgent
from agents.stdfp import STDFPAgent
from agents.trqam import TRQAMAgent
from agents.mdfp import MDFPAgent
from agents.rebrac import ReBRACAgent
from agents.rql import RQLAgent
agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    dfp=DFPAgent,
    dfm=DFMAgent,
    dsrl=DSRLAgent,
    grpo=GRPOAgent,
    svgd=SVGDAgent,
    mfp=MFPAgent,
    qam=QAMAgent,
    stdfp=STDFPAgent,
    trqam=TRQAMAgent,
    mdfp=MDFPAgent,
    rql=RQLAgent,
    rebrac=ReBRACAgent
)
