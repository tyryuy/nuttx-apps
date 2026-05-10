/****************************************************************************
 * apps/examples/executorch_sine/executorch_sine_math.c
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 ****************************************************************************/

#include <stdint.h>

/****************************************************************************
 * Public Functions
 ****************************************************************************/

float nearbyintf(float x)
{
  float ax;
  uint32_t whole;
  float frac;
  int neg;

  if (x != x || x == 0.0f)
    {
      return x;
    }

  neg = x < 0.0f;
  ax = neg ? -x : x;

  if (ax >= 16777216.0f)
    {
      return x;
    }

  whole = (uint32_t)ax;
  frac = ax - (float)whole;

  if (frac > 0.5f || (frac == 0.5f && (whole & 1) != 0))
    {
      whole++;
    }

  return neg ? -(float)whole : (float)whole;
}
