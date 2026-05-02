#include <nuttx/config.h>

#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include <nuttx/sensors/ioctl.h>

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

      /*
       * NuttX lsm6dsl legacy driver の read() は raw 16bit 値を返す。
       * driver の初期設定が ±16g の場合、おおまかに 0.488 mg/LSB。
       */
      printf("accel raw: x=%6d y=%6d z=%6d  approx[mg]: x=%8.1f y=%8.1f z=%8.1f\n",
             xyz[0], xyz[1], xyz[2],
             xyz[0] * 0.488f,
             xyz[1] * 0.488f,
             xyz[2] * 0.488f);

      usleep(200000);
    }

  ioctl(fd, SNIOC_STOP, 0);
  close(fd);
  return 0;
}

