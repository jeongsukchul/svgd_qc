from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent
from agents.dfp import DFPAgent
from agents.grpo import GRPOAgent
from agents.svgd import SVGDAgent
from agents.mfp import MFPAgent
from agents.trqam import TRQAMAgent

agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    dfp=DFPAgent,
    grpo=GRPOAgent,
    svgd=SVGDAgent,
    mfp=MFPAgent,
    trqam=TRQAMAgent,
)
