def assign_seats(passengers, plane):
    """Assign seats, keeping each family together where possible."""
    families = group_by_booking(passengers)
    families.sort(key=lambda f: -len(f))
    seats = plane.seat_map()
    for fam in families:
        block = seats.find_contiguous(len(fam))
        if block is None:
            block = seats.find_scattered(len(fam))
        for p, s in zip(fam, block):
            s.assign(p)
    return seats
