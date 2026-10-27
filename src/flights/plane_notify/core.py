"""Flight-event checks extracted from Jxck-S/plane-notify, GPL-3.0.

Upstream: e1aa8c668896cd6eaa80c42db813eb0487fb02d9, planeClass.py:313-372.
Copyright the plane-notify contributors. See LICENSE.

Detection branches below are copied verbatim except removal of console output
and injection of the nearest-airport lookup. Keep upstream's equality behavior
at 15,000 ft and its persistent landing_plausible state after reacquisition.
The wrapper supplies the replay clock, captures events, and updates history.
"""

from datetime import datetime
from typing import Callable


class PlaneNotify:
    def __init__(self, closest_airport: Callable[[float, float], dict]) -> None:
        self.closest_airport = closest_airport
        self.data_loss_mins = 5
        self.alt_ft = None
        self.below_desired_ft = None
        self.last_below_desired_ft = None
        self.feeding = None
        self.last_feeding = None
        self.on_ground = None
        self.last_on_ground = None
        self.latitude = None
        self.longitude = None
        self.last_pos_datetime = None
        self.landing_plausible = False
        # Upstream initializes these on its first check. Define them for an
        # offline replay that may begin with an empty poll.
        self.tookoff = False
        self.landed = False

    def check(self, now: datetime) -> str | None:
        trigger_type = None
        time_since_contact = (
            None if self.last_pos_datetime is None else now - self.last_pos_datetime
        )
#Check if below desire ft
        desired_ft = 15000
        if self.alt_ft is None or self.alt_ft > desired_ft:
            self.below_desired_ft = False
        elif self.alt_ft < desired_ft:
            self.below_desired_ft = True
#Check if tookoff
        if self.below_desired_ft and self.on_ground is False:
            if self.last_on_ground:
                self.tookoff = True
                trigger_type = "no longer on ground"
                type_header = "Took off from"
            elif self.last_feeding is False and self.feeding and self.landing_plausible == False:
                nearest_airport_dict = self.closest_airport(self.latitude, self.longitude)
                if nearest_airport_dict['elevation_ft'] != "":
                    alt_above_airport = (self.alt_ft - int(nearest_airport_dict['elevation_ft']))
                else:
                    alt_above_airport = None
                if (alt_above_airport != None and alt_above_airport <= 10000) or self.alt_ft <= 15000:
                    self.tookoff = True
                    trigger_type = "data acquisition"
                    type_header = "Took off near"
            else:
                self.tookoff = False
        else:
            self.tookoff = False

#Check if Landed
        if self.on_ground and self.last_on_ground is False and self.last_below_desired_ft:
            self.landed = True
            trigger_type = "now on ground"
            type_header = "Landed in"
            self.landing_plausible = False
        #Set status for landing plausible
        elif self.below_desired_ft and self.last_feeding and self.feeding is False and self.last_on_ground is False:
            self.landing_plausible = True

        elif self.landing_plausible and self.feeding is False and time_since_contact.total_seconds() >= (self.data_loss_mins * 60):
            nearest_airport_dict = self.closest_airport(self.latitude, self.longitude)
            if nearest_airport_dict['elevation_ft'] != "":
                alt_above_airport = (self.alt_ft - int(nearest_airport_dict['elevation_ft']))
            else:
                alt_above_airport = None
            if (alt_above_airport != None and alt_above_airport <= 10000) or self.alt_ft <= 15000:
                self.landing_plausible = False
                self.on_ground = None
                self.landed = True
                trigger_type = "data loss"
                type_header = "Landed near"
            else:
                self.landing_plausible = False
                self.on_ground = None
        else:
            self.landed = False

        self.last_feeding = self.feeding
        self.last_on_ground = self.on_ground
        self.last_below_desired_ft = self.below_desired_ft
        return trigger_type if self.tookoff or self.landed else None
