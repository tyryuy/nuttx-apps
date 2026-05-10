#include <nuttx/config.h>

#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include <nuttx/sensors/ioctl.h>

static void print_mg_x10(int16_t raw)
{
  int32_t mg_x10;

  /*
   * The current legacy LSM6DSL setup uses +/-16g, approximately
   * 0.488 mg/LSB.  Keep the display integer-only because this NuttX config
   * does not enable floating-point printf formatting.
   */

  mg_x10 = ((int32_t)raw * 488 + (raw >= 0 ? 50 : -50)) / 100;

  if (mg_x10 < 0)
    {
      putchar('-');
      mg_x10 = -mg_x10;
    }

  printf("%ld.%01ld", (long)(mg_x10 / 10), (long)(mg_x10 % 10));
}

int main(int argc, char *argv[])
{
  int fd;
  int16_t xyz[3];
  ssize_t nread;
  int i;

  fd = open("/dev/accel0", O_RDONLY);
  if (fd < 0)
    {
      perror("open /dev/accel0");
      return 1;
    }

  if (ioctl(fd, SNIOC_START, 0) < 0)
    {
      perror("SNIOC_START");
      close(fd);
      return 1;
    }

  for (i = 0; i < 20; i++)
    {
      nread = read(fd, xyz, sizeof(xyz));
      if (nread != sizeof(xyz))
        {
          perror("read");
          break;
        }

      printf("accel raw: x=%6d y=%6d z=%6d  approx[mg]: x=",
             xyz[0], xyz[1], xyz[2]);
      print_mg_x10(xyz[0]);
      printf(" y=");
      print_mg_x10(xyz[1]);
      printf(" z=");
      print_mg_x10(xyz[2]);
      printf("\n");

      usleep(200000);
    }

  ioctl(fd, SNIOC_STOP, 0);
  close(fd);
  return 0;
}
