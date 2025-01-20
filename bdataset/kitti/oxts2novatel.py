
# https://docs.novatel.com/OEM7/Content/Logs/BESTPOS.htm
# https://github.com/ApolloAuto/apollo/blob/v5.0.0/modules/drivers/gnss/proto/gnss_best_pose.proto
SOL_TYPES = {
    'NONE': 0,
    'FIXEDPOS': 1,
    'FIXEDHEIGHT': 2,
    'FLOATCONV': 4,
    'WIDELANE': 5,
    'NARROWLANE': 6,
    'DOPPLER_VELOCITY': 8,
    'SINGLE': 16,
    'PSRDIFF': 17,
    'WAAS': 18,
    'PROPAGATED': 19,
    'OMNISTAR': 20,
    'L1_FLOAT': 32,
    'IONOFREE_FLOAT': 33,
    'NARROW_FLOAT': 34,
    'L1_INT': 48,
    'WIDE_INT': 49,
    'NARROW_INT': 50,
    'RTK_DIRECT_INS': 51,
    'INS_SBAS': 52,
    'INS_PSRSP': 53,
    'INS_PSRDIFF': 54,
    'INS_RTKFLOAT': 55,
    'INS_RTKFIXED': 56,
    'INS_OMNISTAR': 57,
    'INS_OMNISTAR_HP': 58,
    'INS_OMNISTAR_XP': 59,
    'OMNISTAR_HP': 64,
    'OMNISTAR_XP': 65,
    'PPP_CONVERGING': 68,
    'PPP': 69,
    'INS_PPP_CONVERGING': 73,
    'INS_PPP': 74
}

# NCOM manual from OxTS
POS_MODES = {
    'None': 0,
    'Search': 1,
    'Doppler': 2,
    'SPS': 3,
    'Differential': 4,
    'RTK Float': 5,
    'RTK Integer': 6,
    'WAAS': 7,
    'OmniSTAR': 8,
    'OmniSTAR HP': 9,
    'No data': 10,
    'Blanked': 11,
    'Doppler (PP)': 12,
    'SPS (PP)': 13,
    'Differential (PP)': 14,
    'RTK Float (PP)': 15,
    'RTK Integer (PP)': 16,
    'OmniSTAR XP': 17,
    'CDGPS': 18,
    'Not recognised': 19,
    'gxDoppler': 20,
    'gxSPS': 21,
    'gxDifferential': 22,
    'gxFloat': 23,
    'gxInteger': 24,
    'ixDoppler': 25,
    'ixSPS': 26,
    'ixDifferential': 27,
    'ixFloat': 28,
    'ixInteger': 29,
    'PPP converging': 30,
    'PPP': 31,
    'Unknown': 32
}

POS_MODE_SOL_TYPE_MAPPING = {
    'None': 'NONE',
    'SPS': 'SINGLE',
    'Differential': 'PSRDIFF',
    'RTK Float': 'L1_FLOAT',
    'RTK Integer': 'NARROW_INT',
    'WAAS': 'WAAS',
    'OmniSTAR': 'OMNISTAR',
    'OmniSTAR HP': 'OMNISTAR_HP',
    'OmniSTAR XP': 'OMNISTAR_XP'
}

def pos_mode2str(pos_mode):
    for k, v in POS_MODES.items():
        if v == pos_mode:
            return k
    else:
        raise RuntimeError("Invalid pos_mode!")

def pos_mode2sol_type(pos_mode):
    pos_mode_str = pos_mode2str(pos_mode)
    if pos_mode_str in POS_MODE_SOL_TYPE_MAPPING.keys():
        sol_type_str = POS_MODE_SOL_TYPE_MAPPING[pos_mode_str]
        return sol_type_str, SOL_TYPES[sol_type_str]
    else:
        raise RuntimeError("No corresponding in Novatel solution types!")

def pos_acc2std_dev(acc):
    return acc
