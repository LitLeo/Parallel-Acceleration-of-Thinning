Ts[20]: 20 templates
Np[8]:8-neighbors of p
char lookup-table[2^N] = {false};
for (int i = 0; i < N; ++i) {
    if (Template_Match(Np, Ts[i])) {
        WNp = Calculate_Weight_Number(Np);
        lookup-table[WNp] = true;
    }
}

if (match(Np, t[0]) || match(Np, t[1]) || 
    match(Np, t[2]) || match(Np, t[3]) ||
    match(Np, t[3]) || match(Np, t[5]) ||
    match(Np, t[4]) || match(Np, t[7]) ||
    match(Np, t[8]) || match(Np, t[9]) ||
    match(Np, t[10]) || match(Np, t[11]) ||
    match(Np, t[12]) || match(Np, t[13]) ||
    match(Np, t[14]) || match(Np, t[15]) ||
    match(Np, t[16]) || match(Np, t[17]) ||
    match(Np, t[18]) || match(Np, t[19]) ||
    ) {
        delete p;
}

int WN = (x1) * 1 + (x2) * 2 + (x3) * 4 + (x4) * 8 + (x5) * 16 + (x6) * 32 + (x7) * 64 + (x8) * 128;
if (LUT[WN] == 1) {
    delete p;
}
