from typing import *


class Fees:

    FeeDivsor = 10000

    def __init__(self, trade_fee: int, admin_fee: int):
        self.trade_fee = trade_fee
        self.admin_fee = admin_fee

    def tradeFee(self, amount: int) -> int:
        return amount * self.trade_fee / Fees.FeeDivsor

    def adminFee(self, amount: int) -> int:
        return amount * self.admin_fee / Fees.FeeDivsor

    def normalized_trade_fee(self, num_coins: int, amount: int) -> int:
        adjusted_trade_fee = int(
            (self.trade_fee * num_coins) / (4 * (num_coins - 1)))
        return amount * adjusted_trade_fee / Fees.FeeDivsor


"""
Calculate invariant D
:param amp: factor A, can get from get_stable_pool interface;
:param c_amounts: vector of c_amounts in the pool, can get from get_stable_pool interface;
:return: invariant D
"""
def calc_d(amp: int, c_amounts: List[int]) -> int:
    n_coins = len(c_amounts)
    sum_amounts = sum(c_amounts)
    # Newton Iteration to resolve equation of higher degree 
    #   previous approaching value
    d_prev = 0
    #   initial approaching value
    d = sum_amounts
    #   Max 256 round approaching iteration
    for i in range(256):
        #   to calc D_prod, as much precise as possible
        d_prod = d
        for c_amount in c_amounts:
            d_prod = d_prod * d / (c_amount * n_coins)
        #   store previous approaching value
        d_prev = d
        #   get cur-round approaching value
        ann = amp * n_coins**n_coins
        # d = (ann * sum_amounts + d_prod * n_coins) * d_prev / ((ann - 1) * d_prev + (n_coins + 1) * d_prod)
        numerator = d_prev * (d_prod * n_coins + ann * sum_amounts)
        denominator = d_prev * (ann - 1) + d_prod * (n_coins + 1)
        d = numerator / denominator
        #   iteration terminating condition
        if abs(d-d_prev) <= 1:
            break

    return d


"""
Calc new Y token amount with a new X token amount(to keep invariant D fixed)
:param amp: factor A, can get from get_stable_pool interface;
:param x_c_amount: X token's new c_amount
:param current_c_amounts: vector of currently c_amounts in the pool, can get from get_stable_pool interface;
:param index_x: Xtoken index, starts from 0
:param index_y: Ytoken index, starts from 0
:return: new Ytoken c_amount
"""
def calc_y(
    amp: int,
    x_c_amount: int,
    current_c_amounts: List[int],
    index_x: int,
    index_y: int
) -> int:
    n_coins = len(current_c_amounts)
    ann = amp * n_coins**n_coins

    # invariant D
    d = calc_d(amp, current_c_amounts)

    # Solve for y by approximating: y**2 + b*y = c
    s_ = x_c_amount
    c = d * d / x_c_amount
    for i in range(n_coins):
        if i != index_x and i != index_y:
            s_ += current_c_amounts[i]
            c = c * d / current_c_amounts[i]
    c = c * d / (ann * n_coins**n_coins)
    b = d / ann + s_

    # Newton Iteration to resolve equation of higher degree   
    y_prev = 0
    y = d
    for i in range(256):
        y_prev = y
        # $ y_{k+1} = \frac{y_k^2 + c}{2y_k + b - D} $
        y_numerator = y**2 + c
        y_denominator = 2 * y + b - d
        y = y_numerator / y_denominator
        #   iteration terminating condition
        if abs(y-y_prev) <= 1:
            break

    return y


"""
Calc initial liquidity
:param amp: factor A, can get from get_stable_pool interface;
:param deposit_c_amounts: vector of depositing token c_amount
:return: [lpt for user, lpt for fee]
"""
def calc_init_liquidity(
    amp: int,
    deposit_c_amounts: List[int],
) -> Tuple[int, int]:
    # n_coins = len(deposit_c_amounts)
    d_0 = calc_d(amp, deposit_c_amounts)
    return (d_0, 0)


"""
Calc add liquidity
:param amp: factor A, can get from get_stable_pool interface;
:param deposit_c_amounts: vector of depositing token c_amount
:param old_c_amounts: vector of currently c_amounts in the pool, can get from get_stable_pool interface;
:param pool_token_supply: currenty supply of lpt
:param fees: (fee ratio in bps, protocol/fee rate in bps)
:return: [lpt for user, lpt for fee]
"""
def calc_add_liquidity_legacy(
    amp: int,
    deposit_c_amounts: List[int],
    old_c_amounts: List[int],
    pool_token_supply: int,
    fees: Fees
) -> Tuple[int, int]:
    n_coins = len(old_c_amounts)
    d_0 = calc_d(amp, old_c_amounts)
    # new vector of token's c_amount in pool after deposit
    c_amounts = [x + y for x, y in zip(old_c_amounts, deposit_c_amounts)]
    # the new D despite fee
    d_1 = calc_d(amp, c_amounts)
    assert(d_1 > d_0)
    # adjust new vector of token's c_amount according to fee policy
    for i in range(n_coins):
        ideal_balance = old_c_amounts[i] * d_1 / d_0
        difference = abs(ideal_balance - c_amounts[i])
        fee = fees.normalized_trade_fee(n_coins, difference)
        c_amounts[i] -= fee
    # the new D under fee impaction
    d_2 = calc_d(amp, c_amounts)

    # it should be d1 >= d2 > d0 and 
    # (d1-d2) => fee part,
    # (d2-d0) => mint_shares for user (fee charged),
    assert(d_1 >= d_2)
    assert(d_2 > d_0)

    # mint user lpt
    mint_shares = pool_token_supply * (d_2 - d_0) / d_0
    # total mint if no fee
    diff_shares = pool_token_supply * (d_1 - d_0) / d_0

    # fee = diff_shares - mint_shares
    return (mint_shares, diff_shares - mint_shares)


"""
Calc remove liquidity by share
:param shares: lpt about to remove from pool
:param c_amounts: vector of currently c_amounts in the pool, can get from get_stable_pool interface;
:param pool_token_supply: currenty supply of lpt
:return: vector of each token's c_amounts that user can get from this removal
"""
def calc_remove_liquidity(
        shares: int,
        c_amounts: List[int],
        pool_token_supply: int) -> List[int]:
    return [x * shares / pool_token_supply for x in c_amounts]


"""
Calc remove liquidity by token combinations
:param amp: factor A, can get from get_stable_pool interface;
:param removed_c_amounts: vector of each token's c_amounts that user want to get from this removal
:param old_c_amounts: vector of currently c_amounts in the pool, can get from get_stable_pool interface;
:param pool_token_supply: currenty supply of lpt
:param fees: (fee ratio in bps, protocol/fee rate in bps)
:return: [lpt user will burn, lpt for fee]
"""
def calc_remove_liquidity_by_tokens(
    amp: int,
    removed_c_amounts: List[int],
    old_c_amounts: List[int],
    pool_token_supply: int,
    fees: Fees
) -> Tuple[int, int]:
    n_coins = len(old_c_amounts)
    d_0 = calc_d(amp, old_c_amounts)
    # new token's c_amount in the pool after this removal
    c_amounts = [x - y for x, y in zip(old_c_amounts, removed_c_amounts)]
    # the new D despite fee
    d_1 = calc_d(amp, c_amounts)
    assert(d_1 < d_0)

    # apply fee policy, got new token's c_amount in the pool
    for i in range(n_coins):
        ideal_balance = old_c_amounts[i] * d_1 / d_0
        difference = abs(ideal_balance - c_amounts[i])
        fee = fees.normalized_trade_fee(n_coins, difference)
        c_amounts[i] -= fee
    # the new D under fee impaction
    d_2 = calc_d(amp, c_amounts)

    # it should be d2 <= d1 < d0 and 
    # (d0-d2) => burn_shares (plus fee),
    # (d0-d1) => diff_shares (without fee),
    # (d1-d2) => fee part,
    assert(d_2 <= d_1)
    assert(d_1 < d_0)

    # lpt that user would burne
    burn_shares = pool_token_supply * (d_0 - d_2) / d_0
    # lpt burned if no fee
    diff_shares = pool_token_supply * (d_0 - d_1) / d_0

    # fee = burn_shares - diff_shares
    return (burn_shares, burn_shares - diff_shares)


"""
Calc swap result (get_return)
:param amp: factor A, can get from get_stable_pool interface;
:param in_token_idx: token in index, starts from 0
:param in_c_amount: depositing token c_amount
:param out_token_idx: token out index, starts from 0
:param old_c_amounts: vector of currently c_amounts in the pool, can get from get_stable_pool interface;
:param fees: (fee ratio in bps, protocol/fee rate in bps)
:return: [swap out token's c_amount, fee c_amount]
"""
def calc_swap(
    amp: int,
    in_token_idx: int,
    in_c_amount: int,
    out_token_idx: int,
    old_c_amounts: List[int],
    fees: Fees
) -> Tuple[int, int]:

    # the new Y token's c_amount
    y = calc_y(amp, in_c_amount +
               old_c_amounts[in_token_idx], old_c_amounts, in_token_idx, out_token_idx)
    # swap out c_amount if no fee
    dy = old_c_amounts[out_token_idx] - y
    if dy > 0:
        # off-by-one issue
        dy = dy - 1
    # apply fee policy
    trade_fee = fees.tradeFee(dy)
    # real swapped out c_amount
    amount_swapped = dy - trade_fee

    return (amount_swapped, trade_fee)


if __name__ == '__main__':

    # View call: v2.ref-finance.near.get_stable_pool({"pool_id": 3020})
    # {
    # token_account_ids: [
    #     'usn',
    #     'dac17f958d2ee523a2206206994597c13d831ec7.factory.bridge.near'
    # ],
    # decimals: [ 18, 6 ],
    # amounts: [ '46985138221057239167795112', '41334577481181' ],
    # c_amounts: [ '46985138221057239167795112', '41334577481181693154941696' ],
    # total_fee: 5,
    # shares_total_supply: '88243780559357930478468647',
    # amp: 240
    # }

    # to get swap info with real fee policy
    (swap_out, fee_part) = calc_swap(
        240,
        0,
        1000000000000000000,
        1,
        [46985138221057239167795112, 41334577481181693154941696],
        Fees(5, 2000)
    )
    print("With fee, 1 unit usn can swap_out: %s, fee_part: %s" % (swap_out, fee_part))


    (swap_out, fee_part) = calc_swap(
        240,
        0,
        1000000000000000000,
        1,
        [46985138221057239167795112, 41334577481181693154941696],
        Fees(0, 0)
    )
    print("Without fee, 1 unit usn can swap_out: %s, fee_part: %s" % (swap_out, fee_part))

    # (minted, fee_part) = calc_init_liquidity(
    #     240,
    #     [123000000000000000000, 123000000000000000000, 123000000000000000000],
    # )
    # print("minted: %s" % minted)
    # print("fee_part: %s" % fee_part)

    # near view $REF_EX predict_add_stable_liquidity '{"pool_id": 79, "amounts": ["100000000", "0", "0"]}'
    # (minted, fee_part) = calc_add_liquidity(
    #     240,
    #     [1000000000000000000, 1000000000000000000, 1000000000000000000],
    #     [123000000000000000000, 123000000000000000000, 123000000000000000000],
    #     369000000000000000000,
    #     Fees(5, 2000)
    #     )
    # print("minted: %s" % minted)
    # print("fee_part: %s" % fee_part)

    # (minted, fee_part) = calc_add_liquidity(
    #     240,
    #     [2000000000000000000, 0, 1000000000000000000],
    #     [123000000000000000000, 123000000000000000000, 123000000000000000000],
    #     369000000000000000000,
    #     Fees(5, 2000)
    #     )
    # print("minted: %s" % minted)
    # print("fee_part: %s" % fee_part)

    # # near view $REF_EX predict_remove_liquidity '{"pool_id": 79, "shares": "100000000000000000000"}'
    # recv_tokens = calc_remove_liquidity(
    #     100000000000000000000,
    #     [23739540956725338048027, 23693993795668851900686, 12799207746046363295015],
    #     60229882460024119712683
    #     )
    # print("recv %s" % recv_tokens)

    # # near view $REF_EX predict_remove_liquidity_by_tokens '{"pool_id": 79, "amounts": ["100000000", "0", "0"]}'
    # (burned, fee_part) = calc_remove_liquidity_by_tokens(
    #     240,
    #     [100000000000000000000, 0, 0],
    #     [23739540956798478700217, 23693993795556946583088, 12799207746085091106157],
    #     60229882460024116126568,
    #     Fees(5, 2000)
    #     )
    # print("burned: %s" % burned)
    # print("fee_part: %s" % fee_part)

    
